import math
from renderers.renderer import *
from renderers.rendererRedner import RendererRedner
from renderers.rendererMitsuba import RendererMitsuba
from renderers.rendererVertexBased import RendererVertexBased
from sphericalharmonics import SphericalHarmonics
from morphablemodel import MorphableModel
from renderers import *
from camera import Camera
from customRenderer import *
from utils import *
import polyscope as ps
class Pipeline:

    def __init__(self,  config, rendererName = ''):
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
        self.renderer = self.createRenderer(rendererName) 
        self.renderer.morphableModel = self.morphableModel
        
        self.uvMap = self.morphableModel.uvMap.clone()
        self.uvMap[:, 1] = 1.0 - self.uvMap[:, 1]
        self.faces32 = self.morphableModel.faces.to(torch.int32).contiguous()
        self.shBands = config.bands
        self.sharedIdentity = False

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

    # def renderRedner(self, cameraVerts = None, diffuseTextures = None, specularTextures = None, roughnessTextures = None, renderAlbedo = False):
    #     '''
    #     ray trace an image given camera vertices and corresponding textures
    #     :param cameraVerts: camera vertices tensor [n, verticesNumber, 3]
    #     :param diffuseTextures: diffuse textures tensor [n, texRes, texRes, 3]
    #     :param specularTextures: specular textures tensor [n, texRes, texRes, 3]
    #     :param roughnessTextures: roughness textures tensor [n, texRes, texRes, 1]
    #     :param renderAlbedo: if True render albedo else ray trace image
    #     :param vertexBased: if True we render by vertex instead of ray tracing
    #     :return: ray traced images [n, resX, resY, 4]
    #     '''
        
    #     if cameraVerts is None :
    #         vertices, diffAlbedo, specAlbedo = self.morphableModel.computeShapeAlbedo(self.vShapeCoeff, self.vExpCoeff, self.vAlbedoCoeff)
    #         cameraVerts = self.camera.transformVertices(vertices, self.vTranslation, self.vRotation)

    #     #compute normals
    #     normals = self.morphableModel.meshNormals.computeNormals(cameraVerts)

    #     if diffuseTextures is None:
    #         diffuseTextures = self.morphableModel.generateTextureFromAlbedo(diffAlbedo)

    #     if specularTextures is None:
    #         specularTextures = self.morphableModel.generateTextureFromAlbedo(specAlbedo)

    #     if roughnessTextures is None:
    #         roughnessTextures  = self.vRoughness

    #     envMaps = self.sh.toEnvMap(self.vShCoeffs)

    #     assert(envMaps.dim() == 4 and envMaps.shape[-1] == 3)
    #     assert (cameraVerts.dim() == 3 and cameraVerts.shape[-1] == 3)
    #     assert (diffuseTextures.dim() == 4 and diffuseTextures.shape[1] == diffuseTextures.shape[2] == self.morphableModel.getTextureResolution() and diffuseTextures.shape[-1] == 3)
    #     assert (specularTextures.dim() == 4 and specularTextures.shape[1] == specularTextures.shape[2] == self.morphableModel.getTextureResolution() and specularTextures.shape[-1] == 3)
    #     assert (roughnessTextures.dim() == 4 and roughnessTextures.shape[1] == roughnessTextures.shape[2] == self.morphableModel.getTextureResolution() and roughnessTextures.shape[-1] == 1)
    #     assert(cameraVerts.shape[0] == envMaps.shape[0])
    #     assert (diffuseTextures.shape[0] == specularTextures.shape[0] == roughnessTextures.shape[0])

    #     scenes = self.rendererRedner.buildScenes(cameraVerts, self.faces32, normals, self.uvMap, diffuseTextures, specularTextures, torch.clamp(roughnessTextures, 1e-20, 10.0), self.vFocals, envMaps)
    #     if renderAlbedo:
    #         images = self.rendererRedner.renderAlbedo(scenes)
    #     else:
    #         images = self.rendererRedner.render(scenes)
                
    #     return images
    
    # def renderVertexBased(self, cameraVerts = None, diffuseAlbedo = None, renderAlbedo= False, lightingOnly=False, interpolation = False ):
    #     '''
    #     render the vertices in an image given camera vertices and corresponding albedos
    #     :param cameraVerts: camera vertices tensor [n, verticesNumber, 3]
    #     :param diffuseAlbedo: diffuse textures tensor [n, texRes, texRes, 3]
    #     :param specularAlbedo: specular textures tensor [n, texRes, texRes, 3]
    #     :return: ray traced images [n, resX, resY, 4]
    #     '''
    #     if cameraVerts is None or diffuseAlbedo is None:
    #         vertices, diffuseAlbedo, _ = self.morphableModel.computeShapeAlbedo(self.vShapeCoeff, self.vExpCoeff, self.vAlbedoCoeff)
    #         cameraVerts = self.camera.transformVertices(vertices, self.vTranslation, self.vRotation)
    #     # compute normals
    #     normals = self.morphableModel.meshNormals.computeNormals(cameraVerts)
    #     assert (cameraVerts.dim() == 3 and cameraVerts.shape[-1] == 3)
    #     # compute colors for vertices
    #     shBasisFunctions = self.sh.preComputeSHBasisFunction(normals, sh_order=8)
    #     # render image                
    #     return self.rendererVertex.render(cameraVerts, normals, diffuseAlbedo, self.vShCoeffs, shBasisFunctions,self.vFocals, renderAlbedo=renderAlbedo, lightingOnly = lightingOnly, interpolation=interpolation)
    
    # def renderMitsuba(self, cameraVerts = None, diffuseTextures = None, specularTextures = None, roughnessTextures = None, renderAlbedo = False,):
    #     '''
    #     ray trace an image given camera vertices and corresponding textures
    #     :param cameraVerts: camera vertices tensor [n, verticesNumber, 3]
    #     :param diffuseTextures: diffuse textures tensor [n, texRes, texRes, 3]
    #     :param specularTextures: specular textures tensor [n, texRes, texRes, 3]
    #     :param roughnessTextures: roughness textures tensor [n, texRes, texRes, 1]
    #     :param renderAlbedo: if True render albedo else ray trace image
    #     :param vertexBased: if True we render by vertex instead of ray tracing
    #     :return: ray traced images [n, resX, resY, 4]
    #     '''
    #     if cameraVerts is None :
    #         vertices, diffAlbedo, specAlbedo = self.morphableModel.computeShapeAlbedo(self.vShapeCoeff, self.vExpCoeff, self.vAlbedoCoeff)
    #         cameraVerts = self.camera.transformVertices(vertices, self.vTranslation, self.vRotation)

    #     #compute normals
    #     normals = self.morphableModel.meshNormals.computeNormals(cameraVerts)

    #     if diffuseTextures is None:
    #         diffuseTextures = self.morphableModel.generateTextureFromAlbedo(diffAlbedo)

    #     if specularTextures is None:
    #         specularTextures = self.morphableModel.generateTextureFromAlbedo(specAlbedo)

    #     if roughnessTextures is None:
    #         roughnessTextures  = self.vRoughness

    #     envMaps = self.sh.toEnvMap(self.vShCoeffs)

    #     assert(envMaps.dim() == 4 and envMaps.shape[-1] == 3)
    #     assert (cameraVerts.dim() == 3 and cameraVerts.shape[-1] == 3)
    #     assert (diffuseTextures.dim() == 4 and diffuseTextures.shape[1] == diffuseTextures.shape[2] == self.morphableModel.getTextureResolution() and diffuseTextures.shape[-1] == 3)
    #     assert (specularTextures.dim() == 4 and specularTextures.shape[1] == specularTextures.shape[2] == self.morphableModel.getTextureResolution() and specularTextures.shape[-1] == 3)
    #     assert (roughnessTextures.dim() == 4 and roughnessTextures.shape[1] == roughnessTextures.shape[2] == self.morphableModel.getTextureResolution() and roughnessTextures.shape[-1] == 1)
    #     assert(cameraVerts.shape[0] == envMaps.shape[0])
    #     assert (diffuseTextures.shape[0] == specularTextures.shape[0] == roughnessTextures.shape[0])

    #     # TODO mitsuba should generate an alpha channel to do loss only on geometry part of picture
    #     img = self.rendererMitsuba.render(cameraVerts, self.faces32, normals, self.uvMap, diffuseTextures, specularTextures, torch.clamp(roughnessTextures, 1e-20, 10.0),self.vFocals[0], envMaps)
      
    #     return img.unsqueeze(0) # add batch dimension
    # generic render method
    def render(self, diffuseTextures = None, specularTextures = None, roughnessTextures = None,  renderAlbedo= False, lightingOnly=False, interpolation = False ):
        
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

        envMap = self.sh.toEnvMap(self.vShCoeffs)

        assert(envMap.dim() == 4 and envMap.shape[-1] == 3)
        assert (cameraVerts.dim() == 3 and cameraVerts.shape[-1] == 3)
        assert (diffuseTextures.dim() == 4 and diffuseTextures.shape[1] == diffuseTextures.shape[2] == self.morphableModel.getTextureResolution() and diffuseTextures.shape[-1] == 3)
        assert (specularTextures.dim() == 4 and specularTextures.shape[1] == specularTextures.shape[2] == self.morphableModel.getTextureResolution() and specularTextures.shape[-1] == 3)
        assert (roughnessTextures.dim() == 4 and roughnessTextures.shape[1] == roughnessTextures.shape[2] == self.morphableModel.getTextureResolution() and roughnessTextures.shape[-1] == 1)
        assert(cameraVerts.shape[0] == envMap.shape[0])
        assert (diffuseTextures.shape[0] == specularTextures.shape[0] == roughnessTextures.shape[0])

        shBasisFunctions = self.sh.preComputeSHBasisFunction(normals, sh_order=8)
        
        return self.renderer.render(cameraVerts, self.faces32, normals, self.uvMap, diffAlbedo, diffuseTextures, specularTextures, torch.clamp(roughnessTextures, 1e-20, 10.0), self.vShCoeffs, shBasisFunctions, self.vFocals[0], envMap,renderAlbedo, lightingOnly, interpolation)
    
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
    
    def compute_visuals(self):
        with torch.no_grad():
            input_img_numpy = 255. * self.input_img.detach().cpu().permute(0, 2, 3, 1).numpy()
            output_vis = self.pred_face * self.pred_mask + (1 - self.pred_mask) * self.input_img
            output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()
            
            if self.gt_lm is not None:
                gt_lm_numpy = self.gt_lm.cpu().numpy()
                pred_lm_numpy = self.pred_lkm.detach().cpu().numpy()
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
    
    def reloadRenderer(self, rendererName):
        # TODO clear cache from previous variable ?
        self.renderer = self.createRenderer(rendererName)
        
    def createRenderer(self, rendererName):
        if rendererName == 'redner':
            return RendererRedner(self.config.rtTrainingSamples, self.config.bounces, self.device, self.config.maxResolution, self.config.maxResolution)
        elif rendererName == 'mitsuba':
            return RendererMitsuba(self.config.rtTrainingSamples, self.config.bounces, self.device, self.config.maxResolution, self.config.maxResolution) 
        elif rendererName == 'vertex':
            return RendererVertexBased(self.device, self.config.maxResolution, self.config.maxResolution) 
        else :
            return Renderer()