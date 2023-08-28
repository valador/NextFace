import drjit as dr
import mitsuba as mi
import torch
from renderers.renderer import Renderer 
from mitsuba.scalar_rgb import Transform4f as T

mi.set_variant('cuda_ad_rgb')
import plugins

class RendererMitsuba(Renderer):

    def __init__(self, samples, bounces, device, screenWidth, screenHeight):
        self.samples = samples
        self.bounces = bounces
        self.device = torch.device(device)
        self.counter = 0
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        mi.set_variant('cuda_ad_rgb')
        mi.LogLevel(1)
        dr.set_log_level(dr.LogLevel.Info)
        dr.set_device(0)
        self.scene = self.buildInitialScene() # init my scene
        
    def buildInitialScene(self):
        """
        generate a placeholder scene where we assign default values that will be updated at runtime
        ** make sure the mitsuba_default folder is properly placed in the output folder
        Returns:
            scene (python dictionnary): mitsuba scene object
        """
        # Create scene
        self.scene = mi.load_dict({
            'type': 'scene',
            'integrator': {
                # 'type': 'direct_reparam' #supposed to be better for visibility discontinuities
                'type':'aov',
                'aovs':'dd.y:depth',
                'my_image':{
                    'type':'direct_reparam'
                }
            },
            'sensor':  {
                'type': 'perspective',
                'fov': 5,
                'to_world': T.look_at(
                            origin=(0, 0, 0),
                            target=(0, 0, 1),
                            up=(0., -1, 0.)
                        ),
                'film': {
                    'type': 'hdrfilm',
                    'width':  self.screenWidth,
                    'height': self.screenHeight,
                    'pixel_format':'rgba',
                    'sample_border':True,
                },
            },
            "mesh":{
                "type": "obj",
                "filename": "./output/mitsuba_default/mesh0.obj",
                "face_normals": True,
                # 'bsdf': {
                #     'type': 'rednermat',
                #     'albedo': {
                #         'type': 'bitmap',
                #         'filename': "./output/mitsuba_default/diffuseMap_0.png"
                #     },                
                #     'roughness':{
                #         'type': 'bitmap',
                #         'filename': "./output/mitsuba_default/roughnessMap_0.png"
                        
                #     },
                #     'specular':{
                #         'type': 'bitmap',
                #         'filename': "./output/mitsuba_default/specularMap_0.png"
                        
                #     }
                # }
                'bsdf': {
                    'type': 'principled',
                    'base_color': {
                        'type': 'bitmap',
                        'filename': "./output/mitsuba_default/diffuseMap_0.png"
                    },                
                    'roughness':{
                        'type': 'bitmap',
                        'filename': "./output/mitsuba_default/roughnessMap_0.png"
                        
                    },
                }
            },
            'light': {
                'type': 'envmap',
                'filename':'./output/mitsuba_default/envMap_0.png'
            }
        })
        return self.scene
    
    # STANDALONE because of wrap_ad
    @dr.wrap_ad(source='torch', target='drjit')
    def render_torch_djit(scene, vertices, indices, normal, uv, diffuseTexture, specularTexture, roughnessTexture, fov, envMap, spp=8, seed=1):
        """
        Standalone function that converts torch computations in mitsuba's drjit computations.
        This wrapper handles the propagation of the gradients directly. 
        We also update the values of our mitsuba scene with the values passed as inputs

        Returns:
           image : tensorX
        """
        params = mi.traverse(scene)
        params["sensor.x_fov"] = fov
        # update mesh params
        params["mesh.vertex_positions"] = dr.ravel(mi.TensorXf(vertices))
        params["mesh.faces"] = dr.ravel(mi.TensorXf(indices))
        params["mesh.vertex_normals"] = dr.ravel(mi.TensorXf(normal))
        params["mesh.vertex_texcoords"] = dr.ravel(mi.TensorXf(uv))
        # update BSDF
        # https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#smooth-diffuse-material-diffuse
        # params["mesh.bsdf.albedo.data"] = mi.TensorXf(diffuseTexture)
        # params["mesh.bsdf.specular.data"] = mi.TensorXf(specularTexture) 
        # params["mesh.bsdf.roughness.data"] = mi.TensorXf(roughnessTexture)
        params["mesh.bsdf.base_color.data"] = mi.TensorXf(diffuseTexture)
        # params["mesh.bsdf.specular.data"] = mi.TensorXf(specularTexture) # principled doesnt support specular textures
        params["mesh.bsdf.roughness.data"] = mi.TensorXf(roughnessTexture)
        #update envMaps
        params["light.data"] = mi.TensorXf(envMap)
        
        params.update() 
        img = mi.render(scene, params, spp=spp, seed=seed, seed_grad=seed+1)
        return img
        
    # overloading this method
    
    def render(self, cameraVertices, indices, normals, uv, diffAlbedo, diffuseTexture, specularTexture, roughnessTexture, shCoeffs, sphericalHarmonics, focals, sensors, renderAlbedo=False, lightingOnly=False, interpolation=False):
        """
        middle function between pytorch and mitsuba, we take the tensor values from our pipeline and give it to our standalone wrapper

        Args:
            vertices (Tensor): _description_
            indices (Tensor): _description_
            normal (Tensor): _description_
            uv (Tensor): _description_
            diffuseTexture (Tensor): _description_
            specularTexture (Tensor): _description_
            roughnessTexture (Tensor): _description_
            focal (Tensor): _description_
            envMap (Tensor): _description_

        Returns:
            image Tensor: the render based on our inputs
        """
        envMap = sphericalHarmonics.toEnvMap(shCoeffs)
        assert(envMap.dim() == 4 and envMap.shape[-1] == 3)
        assert(cameraVertices.shape[0] == envMap.shape[0])
        # assert that our scene has the same amount of sensors as the amount of images
        
        self.fov =  torch.tensor([360.0 * torch.atan(self.screenWidth / (2.0 * focals)) / torch.pi]) # from renderer.py
        
        img =  RendererMitsuba.render_torch_djit(self.scene, cameraVertices.squeeze(0), indices.to(torch.float32), normals.squeeze(0), uv, diffuseTexture.squeeze(0), specularTexture.squeeze(0), roughnessTexture.squeeze(0), self.fov.item(), envMap.squeeze(0),self.samples) # returns a pytorch
        rgb_channels = img[..., :3]
        #debug alpha
        mask_alpha = img[..., 4:]  # only take the last channel ?
        # Create a binary mask based on a condition (e.g., depth_mean > threshold)
        threshold = 0.9 # Adjust the threshold as needed
        depth_mean = torch.mean(mask_alpha, axis=-1, keepdim=True)
        mask_alpha = (depth_mean > threshold).float()
        # Concatenate the RGB channels with the binary mask to create the final image 
        final_image = torch.cat((rgb_channels, mask_alpha), dim=-1).unsqueeze(0)

        return final_image 
        