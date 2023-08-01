import torch
import math
import mitsuba as mi
import drjit as dr
import numpy as np
from mitsuba.scalar_rgb import Transform4f as T

class RendererMitsuba:

    def __init__(self, samples, bounces, device, screenWidth, screenHeight):
        self.samples = samples
        self.bounces = bounces
        self.device = torch.device(device)
        self.counter = 0
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        mi.set_variant('cuda_ad_rgb')
        dr.set_device(0)
        self.scene = self.buildInitialScene() # init my scene
        

    def buildInitialScene(self):
        # Create scene
        self.scene = mi.load_dict({
            'type': 'scene',
            'integrator': {
                # 'type': 'prb'
                'type': 'direct_reparam' #supposed to be better for visibility discontinuities
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
                "filename": "C:/Users/AQ14980/Desktop/repos/NextFace/output/Bikerman.jpg/debug/mesh/debug2_iter1000.obj",
                "face_normals": True,
                'bsdf': {
                    'type': 'principled',
                    'base_color': {
                        'type': 'bitmap',
                        'filename': "C:/Users/AQ14980/Desktop/repos/NextFace/output/Bikerman.jpg/specularMap_0.png"
                    },                
                    'roughness':{
                        'type': 'bitmap',
                        'filename': "C:/Users/AQ14980/Desktop/repos/NextFace/output/sarah.jpg/roughnessMap_0.png"
                    }
                }
            },
            'light': {
                'type': 'envmap',
                'filename':'C:/Users/AQ14980/Desktop/repos/NextFace/output/Bikerman.jpg/envMap_0.png'
            }
        })         
        
        return self.scene
    
    def updateScene(self, vertices, indices, normal, uv, diffuseTexture, specularTexture, roughnessTexture, focal, envMap):
        self.fov =  torch.tensor([360.0 * torch.atan(self.screenWidth / (2.0 * focal)) / torch.pi]) # from renderer.py
        
        params = mi.traverse(self.scene)
        params["sensor.x_fov"] = self.fov.item()
        # update mesh params
        # # -> [1 N 3] -> [N 3]
        params["mesh.vertex_positions"] = dr.ravel(mi.TensorXf(vertices.squeeze(0)))
        # REVIEW could slow down the whole thing
        params["mesh.faces"] = dr.ravel(mi.TensorXf(indices.to(torch.float32)))
        params["mesh.vertex_normals"] = dr.ravel(mi.TensorXf(normal.squeeze(0)))
        params["mesh.vertex_texcoords"] = dr.ravel(mi.TensorXf(uv))
        # reflance data is [ X Y 3] so we convert our diffuseTexture to it 
        # update BSDF
        # https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#smooth-diffuse-material-diffuse
        params["mesh.bsdf.base_color.data"] = mi.TensorXf(diffuseTexture.squeeze(0))
        # params["mesh.bsdf.specular"] = mi.TensorXf(specularTexture.squeeze(0))
        params["mesh.bsdf.roughness.data"] = mi.TensorXf(roughnessTexture.squeeze(0))
        
        #update envMaps
        params["light.data"] = mi.TensorXf(envMap.squeeze(0))
        
        params.update() 
        return self.scene
    # STANDALONE because of wrap_ad
    @dr.wrap_ad(source='torch', target='drjit')
    def render_torch_djit(scene, spp=256, seed=1):
        """take a texture, update the scene and render it. uses a wrap ad for backpropagation and for gradients
        we are adding a mitsuba computations in a pytorch pipeline

        Returns:
           image : tensorX
        """
        scene_params = mi.traverse(scene) # params that should receive gradients !!!
        # dr.enable_grad(scene_params["mesh.vertex_positions"])
        # dr.enable_grad(scene_params["mesh.faces"])
        return mi.render(scene, scene_params, spp=spp, seed=seed, seed_grad=seed+1)
        
    def render(self, scene):
            '''
            render scenes with ray tracing
            :param scene:  mitsuba scene
            :return: ray traced images [n, screenWidth, screenHeight, 4]
            '''
            self.counter += 1
            if scene is None:
                scene = self.scene
            scene_params = mi.traverse(scene) # params that should receive gradients !!!
            
            # return RendererMitsuba.render_torch_djit(scene) # returns a pytorch
            return mi.render(scene, scene_params, spp=256, seed=1, seed_grad=2) # return TensorXf
        

    