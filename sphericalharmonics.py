import torch
import math
import numpy as np
import drjit as dr
'''
code taken and adapted from pyredner
'''

# Code adapted from "Spherical Harmonic Lighting: The Gritty Details", Robin Green
# http://silviojemma.com/public/papers/lighting/spherical-harmonic-lighting.pdf
class SphericalHarmonics:
    def __init__(self, envMapResolution, device):
        self.device = device
        self.setEnvironmentMapResolution(envMapResolution)
    def setEnvironmentMapResolution(self, res):
        res = (res, res)
        self.resolution = res
        uv = np.mgrid[0:res[1], 0:res[0]].astype(np.float32)
        self.theta = torch.from_numpy((math.pi / res[1]) * (uv[1, :, :] + 0.5)).to(self.device)
        self.phi = torch.from_numpy((2 * math.pi / res[0]) * (uv[0, :, :] + 0.5)).to(self.device)

    def smoothSH(self, coeffs, window=6):
        ''' multiply (convolve in sptial domain) the coefficients with a low pass filter.
        Following the recommendation in https://www.ppsloan.org/publications/shdering.pdf
        '''
        smoothed_coeffs = torch.zeros_like(coeffs)
        smoothed_coeffs[:, 0] += coeffs[:, 0]
        smoothed_coeffs[:, 1:1 + 3] += \
            coeffs[:, 1:1 + 3] * math.pow(math.sin(math.pi * 1.0 / window) / (math.pi * 1.0 / window), 4.0)
        smoothed_coeffs[:, 4:4 + 5] += \
            coeffs[:, 4:4 + 5] * math.pow(math.sin(math.pi * 2.0 / window) / (math.pi * 2.0 / window), 4.0)
        smoothed_coeffs[:, 9:9 + 7] += \
            coeffs[:, 9:9 + 7] * math.pow(math.sin(math.pi * 3.0 / window) / (math.pi * 3.0 / window), 4.0)
        return smoothed_coeffs


    def associatedLegendrePolynomial(self, l, m, x):
        pmm = torch.ones_like(x)
        if m > 0:
            somx2 = torch.sqrt((1 - x) * (1 + x))
            fact = 1.0
            for i in range(1, m + 1):
                pmm = pmm * (-fact) * somx2
                fact += 2.0
        if l == m:
            return pmm
        pmmp1 = x * (2.0 * m + 1.0) * pmm
        if l == m + 1:
            return pmmp1
        pll = torch.zeros_like(x)
        for ll in range(m + 2, l + 1):
            pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
            pmm = pmmp1
            pmmp1 = pll
        return pll


    def normlizeSH(self, l, m):
        return math.sqrt((2.0 * l + 1.0) * math.factorial(l - m) / \
                         (4 * math.pi * math.factorial(l + m)))

    def SH(self, l, m, theta, phi):
        if m == 0:
            return self.normlizeSH(l, m) * self.associatedLegendrePolynomial(l, m, torch.cos(theta))
        elif m > 0:
            return math.sqrt(2.0) * self.normlizeSH(l, m) * \
                   torch.cos(m * phi) * self.associatedLegendrePolynomial(l, m, torch.cos(theta))
        else:
            return math.sqrt(2.0) * self.normlizeSH(l, -m) * \
                   torch.sin(-m * phi) * self.associatedLegendrePolynomial(l, -m, torch.cos(theta))
    
    
    
        
    def toEnvMap(self, shCoeffs, smooth = False):
        '''
        create an environment map from given sh coeffs
        :param shCoeffs: float tensor [n, bands * bands, 3]
        :param smooth: if True, the first 3 bands are smoothed
        :return: environment map tensor [n, resX, resY, 3]
        '''
        assert(shCoeffs.dim() == 3 and shCoeffs.shape[-1] == 3)
        envMaps = torch.zeros( [shCoeffs.shape[0], self.resolution[0], self.resolution[1], 3]).to(shCoeffs.device)
        for i in range(shCoeffs.shape[0]):
            envMap =self.constructEnvMapFromSHCoeffs(shCoeffs[i], smooth)
            envMaps[i] = envMap
        return envMaps
    
    def constructEnvMapFromSHCoeffs(self, shCoeffs, smooth = False):

        assert (isinstance(shCoeffs, torch.Tensor) and shCoeffs.dim() == 2 and shCoeffs.shape[1] == 3)

        if smooth:
            smoothed_coeffs = self.smoothSH(shCoeffs.transpose(0, 1), 4)
        else:
            smoothed_coeffs =  shCoeffs.transpose(0, 1) #self.smoothSH(shCoeffs.transpose(0, 1), 4) #smooth the first three bands?

        res = self.resolution

        theta = self.theta
        phi =  self.phi
        result = torch.zeros(res[0], res[1], smoothed_coeffs.shape[0], device=smoothed_coeffs.device)
        bands = int(math.sqrt(smoothed_coeffs.shape[1]))
        i = 0

        for l in range(bands):
            for m in range(-l, l + 1):
                sh_factor = self.SH(l, m, theta, phi)
                result = result + sh_factor.view(sh_factor.shape[0], sh_factor.shape[1], 1) * smoothed_coeffs[:, i]
                i += 1
        result = torch.max(result, torch.zeros(res[0], res[1], smoothed_coeffs.shape[0], device=smoothed_coeffs.device))
        return result
    
    def preComputeSHBasisFunction(self, normals, sh_order):
        """we need a matrix that holds all sh basis function  for our order, it is expensive computations so we save it in this class before

        Args:
            normals ([B, N, 3] tensor): holds the current normals 
            sh_order (int): order of the sh that we are using in our system

        Returns:
            [B, N, (sh_order +1)^2 ]: functions used with vSHCoeffs to get a better color approximation
        """
        numCoeffs = (sh_order + 1) ** 2  # Calculate the number of SH coefficients
       
        # Pre-allocate the tensor to hold SH basis
        Y = torch.empty((normals.shape[0], normals.shape[1], numCoeffs), device=normals.device)
        theta = torch.acos(normals[..., 2:]).squeeze()
        phi = torch.atan2(normals[..., 1:2], normals[..., :1]).squeeze()

        # #instead of for loops we can we can use broadcasting and torch.meshgrid to create a 2D grid of l and m values. 
        # # These can be used to compute the SH basis for each combination of l and m in a vectorized way.
        # # Generate all l, m pairs
        # l_values = torch.arange(sh_order + 1).unsqueeze(1).repeat(1, sh_order + 1).reshape(-1)
        # m_values = torch.arange(-sh_order, sh_order + 1).repeat(sh_order + 1)
        # # Make sure l_values and m_values are in the same device as normals
        # l_values = l_values.to(normals.device)
        # m_values = m_values.to(normals.device)
        """l and m are 1D tensors with 81 and 153 elements respectively. 
        The l tensor contains the degree of each spherical harmonic (which is why it has 81 elements for SH order 8),
        and m contains the order of each spherical harmonic (ranging from -l to +l). 
        This is why m has more elements (153 for SH order 8)."""
        # Generate Y_l_m for all l, m
        # Y = self.vectorizedSH(l_values[:, None, None], m_values[:, None, None], theta[None, ...], phi[None, ...]).squeeze()
        # # Compute SH basis for each l, m and normal
        # element = 0
        for l in range(sh_order+1):
            for m in range(-l, l + 1):
                Y_l_m = self.SH(l, m, theta, phi)
                Y[..., element] = Y_l_m.squeeze()
                element += 1
        return Y
    # def tensorial_factorial(self, n):
    #     return torch.exp(torch.lgamma(n + 1))
    
    # def normlizeVectorizedSH(self, l, m):
    #     # Get number of m values for each l
    #     num_m_values = 2 * l + 1
    #     # Repeat each l for corresponding number of m values
    #     l = torch.tensor([li for li, num in zip(l.tolist(), num_m_values.tolist()) for _ in range(num)], device=l.device)
    #     return torch.sqrt((2.0 * l + 1.0) * self.tensorial_factorial(l - torch.abs(m)) / \
    #                     (4 * math.pi * self.tensorial_factorial(l + torch.abs(m))))


    # def vectorizedSH(self, l, m, theta, phi):
    #     cos_theta = torch.cos(theta)
    #     norm = self.normlizeVectorizedSH(l, m)
    #     Y = torch.zeros_like(l, device=m.device)
    #     Y[m == 0] = norm[m == 0] * self.associatedLegendrePolynomial(l[m == 0], 0, cos_theta[m == 0])
    #     if m.max() > 0:
    #         Y_positive = torch.sqrt(2.0) * norm[m > 0] * torch.cos(m[m > 0] * phi[m > 0]) * self.associatedLegendrePolynomial(l[m > 0], m[m > 0], cos_theta[m > 0])
    #         Y[m > 0] = Y_positive
    #     if m.min() < 0:
    #         Y_negative = torch.sqrt(2.0) * norm[m < 0] * torch.sin(torch.abs(m[m < 0]) * phi[m < 0]) * self.associatedLegendrePolynomial(l[m < 0], torch.abs(m[m < 0]), cos_theta[m < 0])
    #         Y[m < 0] = Y_negative
    #     return Y
    
    