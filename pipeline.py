from sphericalharmonics import SphericalHarmonics
from morphablemodel import MorphableModel
from renderer import Renderer
from camera import Camera
from customRenderer import *
from customRenderer.nvdiffrast import NvidiffrastRenderer
from utils import *
class Pipeline:

    def __init__(self,  config):
        '''
        a pipeline can generate and render textured faces under different camera angles and lighting conditions
        :param config: configuration file used to parameterize the pipeline
        '''
        self.config = config
        self.device = config.device
        self.camera = Camera(self.device)
        self.sh = SphericalHarmonics(config.envMapRes, self.device)

        if self.config.lamdmarksDetectorType == 'fan':
            pathLandmarksAssociation = '/landmark_62.txt'
        elif self.config.lamdmarksDetectorType == 'mediapipe':
            pathLandmarksAssociation = '/landmark_62_mp.txt'
        else:
            raise ValueError(f'lamdmarksDetectorType must be one of [mediapipe, fan] but was {self.config.lamdmarksDetectorType}')

        self.morphableModel = MorphableModel(path = config.path,
                                             textureResolution= config.textureResolution,
                                             trimPca= config.trimPca,
                                             landmarksPathName=pathLandmarksAssociation,
                                             device = self.device
                                             )
        self.renderer = Renderer(config.rtTrainingSamples, 1, self.device)
        self.uvMap = self.morphableModel.uvMap.clone()
        self.uvMap[:, 1] = 1.0 - self.uvMap[:, 1]
        self.faces32 = self.morphableModel.faces.to(torch.int32).contiguous()
        self.shBands = config.bands
        self.sharedIdentity = False
        #DANIEL  To Test 
        # Define Renderer
        # Default settings in facerecon_model.py ~ might need to update them for our specific use case
        focal = 1015
        center=112
        camera_d=10
        use_last_fc=False
        z_near=5
        z_far=15
        fov = 2 * np.arctan(center / focal) * 180 / np.pi      
        use_opengl = True   
        self.NvidiffrastRenderer = NvidiffrastRenderer(rasterize_fov=fov, znear=z_near, zfar=z_far, rasterize_size=int(2 * center), use_opengl=use_opengl)

    def initSceneParameters(self, n, sharedIdentity = False):
        '''
        init pipeline parameters (face shape, albedo, exp coeffs, light and  head pose (camera))
        :param n: the the number of parameters (if negative than the pipeline variables are not allocated)
        :param sharedIdentity: if true, the shape and albedo coeffs are equal to 1, as they belong to the same person identity
        :return:
        '''

        if n <= 0:
            return

        self.sharedIdentity = sharedIdentity
        nShape = 1 if sharedIdentity == True else n

        self.vShapeCoeff = torch.zeros([nShape, self.morphableModel.shapeBasisSize], dtype = torch.float32, device = self.device)
        self.vAlbedoCoeff = torch.zeros([nShape, self.morphableModel.albedoBasisSize], dtype=torch.float32, device=self.device)

        self.vExpCoeff = torch.zeros([n, self.morphableModel.expBasisSize], dtype=torch.float32, device=self.device)
        self.vRotation = torch.zeros([n, 3], dtype=torch.float32, device=self.device)
        self.vTranslation = torch.zeros([n, 3], dtype=torch.float32, device=self.device)
        self.vTranslation[:, 2] = 500.
        self.vRotation[:, 0] = 3.14
        self.vFocals = self.config.camFocalLength * torch.ones([n], dtype=torch.float32, device=self.device)
        self.vShCoeffs = 0.0 * torch.ones([n, self.shBands * self.shBands, 3], dtype=torch.float32, device=self.device)
        self.vShCoeffs[..., 0, 0] = 0.5
        self.vShCoeffs[..., 2, 0] = -0.5
        self.vShCoeffs[..., 1] = self.vShCoeffs[..., 0]
        self.vShCoeffs[..., 2] = self.vShCoeffs[..., 0]

        texRes = self.morphableModel.getTextureResolution()
        self.vRoughness = 0.4 * torch.ones([nShape, texRes, texRes, 1], dtype=torch.float32, device=self.device)

    def computeShape(self):
        '''
        compute shape vertices from the shape and expression coefficients
        :return: tensor of 3d vertices [n, verticesNumber, 3]
        '''

        assert(self.vShapeCoeff is not None and self.vExpCoeff is not None)
        vertices = self.morphableModel.computeShape(self.vShapeCoeff, self.vExpCoeff)
        return vertices

    def transformVertices(self, vertices = None):
        '''
        transform vertices to camera coordinate space
        :param vertices: tensor of 3d vertices [n, verticesNumber, 3]
        :return:  transformed  vertices [n, verticesNumber, 3]
        '''

        if vertices is None:
            vertices = self.computeShape()

        assert(vertices.dim() == 3 and vertices.shape[-1] == 3)
        assert(self.vTranslation is not None and self.vRotation is not None)
        assert(vertices.shape[0] == self.vTranslation.shape[0] == self.vRotation.shape[0])

        transformedVertices = self.camera.transformVertices(vertices, self.vTranslation, self.vRotation)
        return transformedVertices

    def render(self, cameraVerts = None, diffuseTextures = None, specularTextures = None, roughnessTextures = None, renderAlbedo = False, vertexBased = False):
        '''
        ray trace an image given camera vertices and corresponding textures
        :param cameraVerts: camera vertices tensor [n, verticesNumber, 3]
        :param diffuseTextures: diffuse textures tensor [n, texRes, texRes, 3]
        :param specularTextures: specular textures tensor [n, texRes, texRes, 3]
        :param roughnessTextures: roughness textures tensor [n, texRes, texRes, 1]
        :param renderAlbedo: if True render albedo else ray trace image
        :param vertexBased: if True we render by vertex instead of ray tracing
        :return: ray traced images [n, resX, resY, 4]
        '''
        if cameraVerts is None:
            vertices, diffAlbedo, specAlbedo = self.morphableModel.computeShapeAlbedo(self.vShapeCoeff, self.vExpCoeff, self.vAlbedoCoeff)
            cameraVerts = self.camera.transformVertices(vertices, self.vTranslation, self.vRotation)

        #compute normals
        normals = self.morphableModel.meshNormals.computeNormals(cameraVerts)

        if diffuseTextures is None:
            diffuseTextures = self.morphableModel.generateTextureFromAlbedo(diffAlbedo)

        if specularTextures is None:
            specularTextures = self.morphableModel.generateTextureFromAlbedo(specAlbedo)

        if roughnessTextures is None:
            roughnessTextures  = self.vRoughness

        envMaps = self.sh.toEnvMap(self.vShCoeffs)

        assert(envMaps.dim() == 4 and envMaps.shape[-1] == 3)
        assert (cameraVerts.dim() == 3 and cameraVerts.shape[-1] == 3)
        assert (diffuseTextures.dim() == 4 and diffuseTextures.shape[1] == diffuseTextures.shape[2] == self.morphableModel.getTextureResolution() and diffuseTextures.shape[-1] == 3)
        assert (specularTextures.dim() == 4 and specularTextures.shape[1] == specularTextures.shape[2] == self.morphableModel.getTextureResolution() and specularTextures.shape[-1] == 3)
        assert (roughnessTextures.dim() == 4 and roughnessTextures.shape[1] == roughnessTextures.shape[2] == self.morphableModel.getTextureResolution() and roughnessTextures.shape[-1] == 1)
        assert(cameraVerts.shape[0] == envMaps.shape[0])
        assert (diffuseTextures.shape[0] == specularTextures.shape[0] == roughnessTextures.shape[0])

        if vertexBased:
            # gamma ?
            # do we take in account rotation when computing normals ?
            # coeff -> face shape -> face shape rotated -> face vertex (based on camera) -> normals (based on face vertex) -> rotated normals *
            vertexColors = self.computeVertexColor(diffuseTextures, specularTextures, roughnessTextures, normals)
            # face_shape -> self.computeShape() -> vertices (if no camera) or cameraVerts (if  camera)
            images = self.computeVertexImage(cameraVerts, vertexColors)
        else:
            scenes = self.renderer.buildScenes(cameraVerts, self.faces32, normals, self.uvMap, diffuseTextures, specularTextures, torch.clamp(roughnessTextures, 1e-20, 10.0), self.vFocals, envMaps)
            if renderAlbedo:
                images = self.renderer.renderAlbedo(scenes)
            else:
                images = self.renderer.render(scenes)
                
        return images

    def landmarkLoss(self, cameraVertices, landmarks, focals, cameraCenters,  debugDir = None):
        '''
        calculate scalar loss between vertices in camera space and 2d landmarks pixels
        :param cameraVertices: 3d vertices [n, nVertices, 3]
        :param landmarks: 2d corresponding pixels [n, nVertices, 2]
        :param landmarks: camera focals [n]
        :param cameraCenters: camera centers [n, 2
        :param debugDir: if not none save landmarks and vertices to an image file
        :return: scalar loss (float)
        '''
        assert (cameraVertices.dim() == 3 and cameraVertices.shape[-1] == 3)
        assert (focals.dim() == 1)
        assert(cameraCenters.dim() == 2 and cameraCenters.shape[-1] == 2)
        assert (landmarks.dim() == 3 and landmarks.shape[-1] == 2)
        assert cameraVertices.shape[0] == landmarks.shape[0] == focals.shape[0] == cameraCenters.shape[0]

        headPoints = cameraVertices[:, self.morphableModel.landmarksAssociation]
        assert (landmarks.shape[-2] == headPoints.shape[-2])

        projPoints = focals.view(-1, 1, 1) * headPoints[..., :2] / headPoints[..., 2:]
        projPoints += cameraCenters.unsqueeze(1)
        loss = torch.norm(projPoints - landmarks, 2, dim=-1).pow(2).mean()
        if debugDir:
            for i in range(projPoints.shape[0]):
                image = saveLandmarksVerticesProjections(self.inputImage.tensor[i], projPoints[i], self.landmarks[i])
                cv2.imwrite(debugDir + '/lp' +  str(i) +'.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        return loss
    
    #WIP
    # Generate colors for each vertices
    def computeVertexColor(self, diffuseTexture, specularTexture, roughnessTexture, normals, gamma = None):
        """
        Return:
            face_color       -- torch.tensor, size (B, N, 3), range (0, 1.)

        Parameters:
            face_texture_diffuse     -- torch.tensor, size (B, N, 3), from texture model, range (0, 1.)
            face_texture_specular     -- torch.tensor, size (B, N, 3), from texture model, range (0, 1.)
            face_texture_roughness     -- torch.tensor, size (B, N, 3), from texture model, range (0, 1.)
            normals        -- torch.tensor, size (B, N, 3), rotated face normal
            gamma            -- torch.tensor, size (B, 27), SH coeffs
        """
        # v_num = face_texture.shape[1]
        a,c = self.sh.a, self.sh.c
        # we init gamma with a bunch of zeros (could cause wrong results)
        gamma = torch.zeros(normals.shape[0],27)
        # Move gamma to the same device as normals
        gamma = gamma.to(normals.device)
        # gamma = torch.ones(normals.shape[0],27)
        batch_size = gamma.shape[0]        
        # default initiation in faceRecon3D (may need to be update or found)
        gamma = gamma.reshape([batch_size, 3, 9])
        init_lit = torch.tensor([0.8, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device=normals.device)
        gamma = gamma + init_lit # use init_lit
        gamma = gamma.permute(0, 2, 1)
        
        Y = torch.cat([
             a[0] * c[0] * torch.ones_like(normals[..., :1]).to(self.device),
            -a[1] * c[1] * normals[..., 1:2],
             a[1] * c[1] * normals[..., 2:],
            -a[1] * c[1] * normals[..., :1],
             a[2] * c[2] * normals[..., :1] * normals[..., 1:2],
            -a[2] * c[2] * normals[..., 1:2] * normals[..., 2:],
            0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * normals[..., 2:] ** 2 - 1),
            -a[2] * c[2] * normals[..., :1] * normals[..., 2:],
            0.5 * a[2] * c[2] * (normals[..., :1] ** 2  - normals[..., 1:2] ** 2)
        ], dim=-1)
        r = Y @ gamma[..., :1]
        g = Y @ gamma[..., 1:2]
        b = Y @ gamma[..., 2:]
        # WIP not sure about that one 
        # bd = face_texture_diffuse 
        # bs = face_texture_roughness * face_texture_specular
        print(gamma.shape)
        print(Y.shape)
        print(r.shape)
        print(g.shape)
        print(b.shape)
        print("normals") # should be B N 3        
        print(normals.shape) # should be B N 3        
        print(diffuseTexture.shape) # should be B N 3 but is B x y 3
        # can i convert texture from B x,y 3 to be linear ? where N is vertex position
        N = normals.shape[1]
        print(N)
        # Reshape diffuseTexture to [1, 262144, 3]
        # Resize diffuseTexture_reshaped to match the size of Y
        #[ 1, x, N, 3]
        diffuseTexture_resized = torch.nn.functional.interpolate(diffuseTexture, size=(N, 3), mode='bilinear')
        # [1, N, 3]
        diffuseTexture_resized = torch.squeeze(diffuseTexture_resized, dim=1)
        print(diffuseTexture_resized.shape) # should be B N 3 but is B x y 3
        
        face_color = torch.cat([r, g, b], dim=-1) * diffuseTexture_resized  
        # we need to update face_color to match Images
        return face_color
    # predict face and mask
    def computeVertexImage(self, vertices, verticesColor) : 
        #pred mask + pred face 
        # Renderer(vertex, face_buffer, face_color)    
        """
        Return:
            predicted mask :
            _ :
            predicted face : 
        Parameters:
            predicted_vertex
            face_buf : # vertex indices for each face. starts from 0. [F,3]         self.face_buf = model['tri'].astype(np.int64) - 1 (based on face model)
            predicted_color : color at each vertices
        """
        # dans saveObj, le visage = faces32 et le nombre de triangles = shape[0]
        
        #TODO do custom projection here to compute an image
        
        return None
    # draw the visuals
    def compute_visuals(self):
        with torch.no_grad():
            input_img_numpy = 255. * self.input_img.detach().cpu().permute(0, 2, 3, 1).numpy()
            output_vis = self.pred_face * self.pred_mask + (1 - self.pred_mask) * self.input_img
            output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()
            
            if self.gt_lm is not None:
                gt_lm_numpy = self.gt_lm.cpu().numpy()
                pred_lm_numpy = self.pred_lm.detach().cpu().numpy()
                output_vis_numpy = self.draw_landmarks(output_vis_numpy_raw, gt_lm_numpy, 'b')
                output_vis_numpy = self.draw_landmarks(output_vis_numpy, pred_lm_numpy, 'r')
            
                output_vis_numpy = np.concatenate((input_img_numpy, 
                                    output_vis_numpy_raw, output_vis_numpy), axis=-2)
            else:
                output_vis_numpy = np.concatenate((input_img_numpy, 
                                    output_vis_numpy_raw), axis=-2)

            self.output_vis = torch.tensor(
                    output_vis_numpy / 255., dtype=torch.float32
                ).permute(0, 3, 1, 2).to(self.device)
    def draw_landmarks(img, landmark, color='r', step=2):
        """
        Return:
            img              -- numpy.array, (B, H, W, 3) img with landmark, RGB order, range (0, 255)
            

        Parameters:
            img              -- numpy.array, (B, H, W, 3), RGB order, range (0, 255)
            landmark         -- numpy.array, (B, 68, 2), y direction is opposite to v direction
            color            -- str, 'r' or 'b' (red or blue)
        """
        if color =='r':
            c = np.array([255., 0, 0])
        else:
            c = np.array([0, 0, 255.])

        _, H, W, _ = img.shape
        img, landmark = img.copy(), landmark.copy()
        landmark[..., 1] = H - 1 - landmark[..., 1]
        landmark = np.round(landmark).astype(np.int32)
        for i in range(landmark.shape[1]):
            x, y = landmark[:, i, 0], landmark[:, i, 1]
            for j in range(-step, step):
                for k in range(-step, step):
                    u = np.clip(x + j, 0, W - 1)
                    v = np.clip(y + k, 0, H - 1)
                    for m in range(landmark.shape[0]):
                        img[m, v[m], u[m]] = c
        return img