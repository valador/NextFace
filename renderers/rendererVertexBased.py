import torch
from renderers.renderer import Renderer 

class RendererVertexBased(Renderer):
    def __init__(self, device, screenWidth = 256, screenHeight = 256):
        self.samples = 8
        self.device = torch.device(device)
        self.near = 0.1
        self.far = 100
        self.upVector = torch.tensor([0.0, -1.0, 0.0])
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        self.gammaCorrection = 2.2
    
    # Generate colors for each vertices
    def computeVertexColor(self, diffAlbedo, shCoeffs, shBasisFunctions, renderAlbedo=False, lightingOnly=False):
        """
        Return:
            face_color       -- torch.tensor, size (B, N, 3), range (0, 1.)

        Parameters:
            diffAlbedo     -- torch.tensor, size (B, N, 3) 
            normals        -- torch.tensor, size (B, N, 3), rotated face normal
            Y              -- torch.tensor, size (B, N, 81), sh basis functions, should be (order +1 )^2
        """
        if renderAlbedo :
            return diffAlbedo
        
        # shCoeffs is of shape [1, 81, 3]
        r = shBasisFunctions @ shCoeffs[..., :1]
        g = shBasisFunctions @ shCoeffs[..., 1:2]
        b = shBasisFunctions @ shCoeffs[..., 2:]
        
        if lightingOnly:
            face_color = torch.cat([r, g, b], dim=-1)
        else:
            face_color = diffAlbedo * torch.cat([r, g, b], dim=-1)
            face_color = torch.clamp(face_color, min=1e-8)
        
        # gamma correction
        return torch.pow(face_color, 1.0/ self.gammaCorrection)
    
    # predict face and mask
    def computeVertexImage(self, cameraVertices, verticesColor, normals, focals, interpolation=False) : 
        """generate multiple vertex based images

        Args:
            cameraVertices (tensor [B, N, 3]): _description_
            verticesColor (tensor [B, N, 3]): _description_
            normals (tensor [B, N, 3]): _description_
            focals (tensor [B]): _description_
            interpolation (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        
        B = cameraVertices.shape[0]  # Batch size
        width = self.screenWidth
        height = self.screenHeight
        images_data = torch.zeros((B, height, width, 4), dtype=torch.float32, device=self.device)
        alpha_channel = torch.zeros((B, width, height, 1)).to(self.device)
        counter = torch.zeros((B, height, width, 3), dtype=torch.float32, device=self.device)
        for i in range(B):
            fov = torch.tensor([360.0 * torch.atan(width / (2.0 * focals[i])) / torch.pi]) # from renderer.py

            # Same code as before, but adapted to handle individual images within the batch
            projMatrix = self.perspectiveProjMatrix(fov, width / height).to(self.device)
            homogeneous_vertices = torch.cat((cameraVertices[i:i+1], torch.ones(1, cameraVertices.shape[1], 1).to(self.device)), -1)
            vertices_in_clip_space = (projMatrix @ homogeneous_vertices.transpose(-1, -2)).transpose(-1, -2)

            vertices_in_clip_space = vertices_in_clip_space[..., :3] / vertices_in_clip_space[..., 3:]
            normal_mask = normals[i:i+1, :, 2] <= 0
            mask = (torch.abs(vertices_in_clip_space[:, :, :2]) <= 1.0).all(dim=-1) & normal_mask
            vertices_in_clip_space = vertices_in_clip_space[mask]

            vertices_in_screen_space = vertices_in_clip_space.clone()
            vertices_in_screen_space[..., 0] = (vertices_in_clip_space[..., 0] + 1) / 2 * width
            vertices_in_screen_space[..., 1] = (vertices_in_clip_space[..., 1] + 1) / 2 * height

            verticesColor_single = verticesColor[i:i+1].squeeze(0)
            mask = mask.squeeze(0)
            colors_in_screen_space = verticesColor_single[mask]

            # Interpolation if needed
            if interpolation:
                vertices_in_screen_space = vertices_in_screen_space.long()  # Convert to long for indexing
                vertices_in_screen_space.clamp_(0, max=self.screenWidth-1)  # Clamp to valid pixel range 
                y_indices, x_indices = vertices_in_screen_space[:, 1], vertices_in_screen_space[:, 0] # create two tensors for values
                # Perform scatter operation + add alpha values
                images_data[i, y_indices, x_indices, :3] += colors_in_screen_space # add colors to pixels
                    # ----------
                # Define the interpolation factor
                interpolation_factor = 32
                # Prepare for bilinear interpolation by adding an extra dimension (required for F.interpolate)
                # Rearrange the tensor to [batch, channel, height, width]
                images_data = images_data.permute(0, 3, 1, 2)  # shape: [Batch, Channels, Height, Width]

                # Perform bilinear interpolation
                images_data = torch.nn.functional.interpolate(images_data, scale_factor=interpolation_factor, mode='bilinear', align_corners=True)

                # Reduce back to the original size by average pooling
                images_data = torch.nn.functional.avg_pool2d(images_data, interpolation_factor)

                # Rearrange the tensor back to [batch, height, width, channel]
                images_data = images_data.permute(0, 2, 3, 1)  # shape: [1, H, W, 3]
            else:
                # Convert vertices and colors to an image without interpolation
                vertices_in_screen_space = vertices_in_screen_space.long()  # Convert to long for indexing
                vertices_in_screen_space.clamp_(0, max=self.screenWidth-1)  # Clamp to valid pixel range 
                y_indices, x_indices = vertices_in_screen_space[:, 1], vertices_in_screen_space[:, 0] # create two tensors for values
                # Perform scatter operation + add alpha values
                counter[i, y_indices, x_indices] += 1 # count all the pixels that have a vertex on them
                images_data[i, y_indices, x_indices, :3] += colors_in_screen_space # add colors to pixels
                
            # Update the alpha channel of those pixels to 1 where we have updated the color
            alpha_mask = counter[i, y_indices, x_indices].sum(dim=1) > 0 # create a mask for each vertex where there is at least one pixel splat there
            alpha_channel[i, y_indices[alpha_mask], x_indices[alpha_mask]] = 1.0 # put a 1.0 to all the pixels that are in our mask
            # Average the color and clamp at 1
        images_data[..., :3] /= counter.clamp(min=1) # average tje color and clamp to one so that we dont divide by one
        images_data = images_data.clamp(0, 1) # clamp values of color and alpha to be between 0 and 1

        # Add alpha channel to the images_data
        images_data[..., 3:] = alpha_channel 
       
        return images_data
    
    def perspectiveProjMatrix(self, fov, aspect_ratio):
        """
        Create a perspective projection matrix.
        
        :param fov: field of view angle in the y direction (in degrees).
        :param aspect_ratio: aspect ratio of the viewport (width/height).
        :param near: distance to the near clipping plane.
        :param far: distance to the far clipping plane.
        """
        f = 1.0 / torch.tan(torch.deg2rad(fov) / 2.0)
        # right handed matrix
        projMatrix = torch.tensor([
            [f / aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, self.far/(self.far- self.near), -(self.far*self.near)/(self.far-self.near)],
            [0, 0, 1, 0]
        ])
        # or flip on x
        # projMatrix = torch.tensor([
        #     [-f / aspect_ratio, 0, 0, 0],
        #     [0, f, 0, 0],
        #     [0, 0, far/(far-near), -(far*near)/(far-near)],
        #     [0, 0, 1, 0]
        # ])
        # or flip on y axis
        # projMatrix = torch.tensor([
        #     [f / aspect_ratio, 0, 0, 0],
        #     [0, -f, 0, 0],
        #     [0, 0, far/(far-near), -(far*near)/(far-near)],
        #     [0, 0, 1, 0]
        # ])
        # or flip on z axis
        # projMatrix = torch.tensor([
        #     [f / aspect_ratio, 0, 0, 0],
        #     [0, f, 0, 0],
        #     [0, 0, -far/(far-near), (far*near)/(far-near)],
        #     [0, 0, -1, 0]
        # ])
        # left handed
        # projMatrix = torch.tensor([
        #     [f / aspect_ratio, 0, 0, 0],
        #     [0, -f, 0, 0],
        #     [0, 0, -far/(far-near), -(far*near)/(far-near)],
        #     [0, 0, -1, 0]
        # ])
        return projMatrix
    # overloading this method
    def render(self, cameraVertices, indices, normals, uv, diffAlbedo, diffuseTexture, specularTexture, roughnessTexture, shCoeffs, sphericalHarmonics, focals, renderAlbedo=False, lightingOnly=False, interpolation=False ):
        """take inputs, generate color for each vertex and project them on the screen

        Args:
            cameraVertices (_type_): _description_
            diffAlbedo (_type_): _description_
            shCoeffs (_type_): _description_
            shBasisFunctions (_type_): _description_
            focals (_type_): _description_
            renderAlbedo (bool, optional): _description_. Defaults to False.
            lightingOnly (bool, optional): _description_. Defaults to False.
            interpolation (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        shBasisFunctions = sphericalHarmonics.preComputeSHBasisFunction(normals, sh_order=8)

        vertexColors = self.computeVertexColor(diffAlbedo, shCoeffs, shBasisFunctions, renderAlbedo=renderAlbedo, lightingOnly=lightingOnly)
        imgs =  self.computeVertexImage(cameraVertices, vertexColors, normals, focals, interpolation=interpolation)
        return imgs
        