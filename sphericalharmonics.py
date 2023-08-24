import torch
import math
import numpy as np
'''
code taken and adapted from pyredner
'''

# Code adapted from "Spherical Harmonic Lighting: The Gritty Details", Robin Green
# http://silviojemma.com/public/papers/lighting/spherical-harmonic-lighting.pdf
class SphericalHarmonics:
    def __init__(self, envMapResolution, device):
        self.device = device
        self.setEnvironmentMapResolution(envMapResolution)
        # DANIEL a and c are defined in the problem by itself, this value are not necessarily true
        self.a = [np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)]
        self.c = [1/np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)]
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
        # if self.Y is not None:
        #     return self.Y
        
        numCoeffs = (sh_order + 1) ** 2  # Calculate the number of SH coefficients
       
        # Pre-allocate the tensor to hold SH basis
        Y = torch.empty((normals.shape[0], normals.shape[1], numCoeffs), device=normals.device)
        theta = torch.acos(normals[..., 2:]).squeeze()
        phi = torch.atan2(normals[..., 1:2], normals[..., :1]).squeeze()

        # Compute SH basis for each l, m and normal
        element = 0
        for l in range(sh_order+1):
            for m in range(-l, l + 1):
                Y_l_m = self.SH(l, m, theta, phi)
                Y[..., element] = Y_l_m.squeeze()
                element += 1
        self.Y = Y
        return Y
        # return self.sh_eval_9(normals)
    
    
    # take drjit method in our code
    def sh_eval_9(self, d) :
        # assert (d.shape == 3, "The parameter 'd' should be a 3D vector.")
    
        x = d[:, :, 0]
        y = d[:, :, 1]
        z = d[:, :, 2]
        z2 = z * z
        out = torch.zeros(81).to(self.device)
        out[0] = 0.28209479177387814
        out[2] = z * 0.488602511902919923
        out[6] = self.fmadd(z2, 0.94617469575756008, -0.315391565252520045)
        out[12] = z * self.fmadd(z2, 1.865881662950577, -1.1195289977703462)
        out[20] = self.fmadd(z * 1.98431348329844304, out[12], out[6] * -1.00623058987490532)
        out[30] = self.fmadd(z * 1.98997487421323993, out[20], out[12] * -1.00285307284481395)
        out[42] = self.fmadd(z * 1.99304345718356646, out[30], out[20] * -1.00154202096221923)
        out[56] = self.fmadd(z * 1.99489143482413467, out[42], out[30] * -1.00092721392195827)
        out[72] = self.fmadd(z * 1.9960899278339137, out[56], out[42] * -1.00060078106951478)
        out[90] = self.fmadd(z * 1.99691119506793657, out[72], out[56] * -1.0004114379931337)
        c0 = x;
        s0 = y;

        tmp_a = -0.488602511902919978
        out[3] = tmp_a * c0;
        out[1] = tmp_a * s0;
        tmp_b = z * -1.09254843059207896
        out[7] = tmp_b * c0;
        out[5] = tmp_b * s0;
        tmp_c = self.fmadd(z2, -2.28522899732232876, 0.457045799464465774)
        out[13] = tmp_c * c0;
        out[11] = tmp_c * s0;
        tmp_a = z * self.fmadd(z2, -4.6833258049010249, 2.00713963067186763)
        out[21] = tmp_a * c0;
        out[19] = tmp_a * s0;
        tmp_b = self.fmadd(z * 2.03100960115899021, tmp_a, tmp_c * -0.991031208965114985)
        out[31] = tmp_b * c0;
        out[29] = tmp_b * s0;
        tmp_c = self.fmadd(z * 2.02131498923702768, tmp_b, tmp_a * -0.995226703056238504)
        out[43] = tmp_c * c0;
        out[41] = tmp_c * s0;
        tmp_a = self.fmadd(z * 2.01556443707463773, tmp_c, tmp_b * -0.99715504402183186)
        out[57] = tmp_a * c0;
        out[55] = tmp_a * s0;
        tmp_b = self.fmadd(z * 2.01186954040739119, tmp_a, tmp_c * -0.998166817890174474)
        out[73] = tmp_b * c0;
        out[71] = tmp_b * s0;
        tmp_c = self.fmadd(z * 2.00935312974101166, tmp_b, tmp_a * -0.998749217771908837)
        out[91] = tmp_c * c0;
        out[89] = tmp_c * s0;
        c1 = self.fmsub(x, c0, y * s0)
        s1 = self.fmadd(x, s0, y * c0)

        tmp_a = 0.546274215296039478
        out[8] = tmp_a * c1;
        out[4] = tmp_a * s1;
        tmp_b = z * 1.44530572132027735
        out[14] = tmp_b * c1;
        out[10] = tmp_b * s1;
        tmp_c = self.fmadd(z2, 3.31161143515146028, -0.473087347878779985)
        out[22] = tmp_c * c1;
        out[18] = tmp_c * s1;
        tmp_a = z * self.fmadd(z2, 7.19030517745998665, -2.39676839248666207)
        out[32] = tmp_a * c1;
        out[28] = tmp_a * s1;
        tmp_b = self.fmadd(z * 2.11394181566096995, tmp_a, tmp_c * -0.973610120462326756)
        out[44] = tmp_b * c1;
        out[40] = tmp_b * s1;
        tmp_c = self.fmadd(z * 2.08166599946613307, tmp_b, tmp_a * -0.984731927834661791)
        out[58] = tmp_c * c1;
        out[54] = tmp_c * s1;
        tmp_a = self.fmadd(z * 2.06155281280883029, tmp_c, tmp_b * -0.990337937660287326)
        out[74] = tmp_a * c1;
        out[70] = tmp_a * s1;
        tmp_b = self.fmadd(z * 2.04812235835781919, tmp_a, tmp_c * -0.993485272670404207)
        out[92] = tmp_b * c1;
        out[88] = tmp_b * s1;
        c0 = self.fmsub(x, c1, y * s1)
        s0 = self.fmadd(x, s1, y * c1)

        tmp_a = -0.590043589926643519
        out[15] = tmp_a * c0;
        out[9] = tmp_a * s0;
        tmp_b = z * -1.77013076977993067
        out[23] = tmp_b * c0;
        out[17] = tmp_b * s0;
        tmp_c = self.fmadd(z2, -4.40314469491725369, 0.48923829943525049)
        out[33] = tmp_c * c0;
        out[27] = tmp_c * s0;
        tmp_a = z * self.fmadd(z2, -10.1332578546641603, 2.76361577854477058)
        out[45] = tmp_a * c0;
        out[39] = tmp_a * s0;
        tmp_b = self.fmadd(z * 2.20794021658196149, tmp_a, tmp_c * -0.95940322360024699)
        out[59] = tmp_b * c0;
        out[53] = tmp_b * s0;
        tmp_c = self.fmadd(z * 2.15322168769582012, tmp_b, tmp_a * -0.975217386560017774)
        out[75] = tmp_c * c0;
        out[69] = tmp_c * s0;
        tmp_a = self.fmadd(z * 2.11804417118980526, tmp_c, tmp_b * -0.983662844979209416)
        out[93] = tmp_a * c0;
        out[87] = tmp_a * s0;
        c1 = self.fmsub(x, c0, y * s0)
        s1 = self.fmadd(x, s0, y * c0)

        tmp_a = 0.625835735449176256
        out[24] = tmp_a * c1;
        out[16] = tmp_a * s1;
        tmp_b = z * 2.07566231488104114
        out[34] = tmp_b * c1;
        out[26] = tmp_b * s1;
        tmp_c = self.fmadd(z2, 5.55021390801596581, -0.504564900728724064)
        out[46] = tmp_c * c1;
        out[38] = tmp_c * s1;
        tmp_a = z * self.fmadd(z2, 13.4918050467267694, -3.11349347232156193)
        out[60] = tmp_a * c1;
        out[52] = tmp_a * s1;
        tmp_b = self.fmadd(z * 2.30488611432322132, tmp_a, tmp_c * -0.948176387355465389)
        out[76] = tmp_b * c1;
        out[68] = tmp_b * s1;
        tmp_c = self.fmadd(z * 2.22917715070623501, tmp_b, tmp_a * -0.967152839723182112)
        out[94] = tmp_c * c1;
        out[86] = tmp_c * s1;
        c0 = self.fmsub(x, c1, y * s1)
        s0 = self.fmadd(x, s1, y * c1)

        tmp_a = -0.656382056840170258
        out[35] = tmp_a * c0;
        out[25] = tmp_a * s0;
        tmp_b = z * -2.3666191622317525
        out[47] = tmp_b * c0;
        out[37] = tmp_b * s0;
        tmp_c = self.fmadd(z2, -6.7459025233633847, 0.518915578720260395)
        out[61] = tmp_c * c0;
        out[51] = tmp_c * s0;
        tmp_a = z * self.fmadd(z2, -17.2495531104905417, 3.44991062209810817)
        out[77] = tmp_a * c0;
        out[67] = tmp_a * s0;
        tmp_b = self.fmadd(z * 2.40163634692206163, tmp_a, tmp_c * -0.939224604204370817)
        out[95] = tmp_b * c0;
        out[85] = tmp_b * s0;
        c1 = self.fmsub(x, c0, y * s0)
        s1 = self.fmadd(x, s0, y * c0)

        tmp_a = 0.683184105191914415
        out[48] = tmp_a * c1;
        out[36] = tmp_a * s1;
        tmp_b = z * 2.64596066180190048
        out[62] = tmp_b * c1;
        out[50] = tmp_b * s1;
        tmp_c = self.fmadd(z2, 7.98499149089313942, -0.532332766059542606)
        out[78] = tmp_c * c1;
        out[66] = tmp_c * s1;
        tmp_a = z * self.fmadd(z2, 21.3928901909086377, -3.77521591604270101)
        out[96] = tmp_a * c1;
        out[84] = tmp_a * s1;
        c0 = self.fmsub(x, c1, y * s1)
        s0 = self.fmadd(x, s1, y * c1)

        tmp_a = -0.707162732524596271
        out[63] = tmp_a * c0;
        out[49] = tmp_a * s0;
        tmp_b = z * -2.91570664069931995
        out[79] = tmp_b * c0;
        out[65] = tmp_b * s0;
        tmp_c = self.fmadd(z2, -9.26339318284890467, 0.544905481344053255)
        out[97] = tmp_c * c0;
        out[83] = tmp_c * s0;
        c1 = self.fmsub(x, c0, y * s0)
        s1 = self.fmadd(x, s0, y * c0)

        tmp_a = 0.728926660174829988
        out[80] = tmp_a * c1;
        out[64] = tmp_a * s1;
        tmp_b = z * 3.17731764895469793
        out[98] = tmp_b * c1;
        out[82] = tmp_b * s1;
        c0 = self.fmsub(x, c1, y * s1)
        s0 = self.fmadd(x, s1, y * c1)

        tmp_c = -0.74890095185318839
        out[99] = tmp_c * c0;
        out[81] = tmp_c * s0;
    
    def fmadd(self, a, b, c):
        return (a * b) + c

    def fmsub(self, a, b, c):
        return (a * b) - c
    