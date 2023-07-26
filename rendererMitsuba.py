import torch
import math
import mitsuba as mi
from mitsuba.scalar_rgb import Transform4f as T

import drjit as dr
# def rayTrace(scene,
#              channels,
#              max_bounces = 1,
#              sampler_type = pyredner.sampler_type.sobol,
#              num_samples = 8,
#              seed = None,
#              sample_pixel_center = False,
#              device  = None):
#     if device is None:
#         device = pyredner.get_device()

#     assert(isinstance(scene, list))
#     if seed == None:
#         # Randomly generate a list of seed
#         seed = []
#         for i in range(len(scene)):
#             seed.append(random.randint(0, 16777216))
#     assert(len(seed) == len(scene))
#     # Render each scene in the batch and stack them together
#     imgs = []
#     for sc, se in zip(scene, seed):
#         scene_args = pyredner.RenderFunction.serialize_scene(\
#             scene = sc,
#             num_samples = num_samples,
#             max_bounces = max_bounces,
#             sampler_type = sampler_type,
#             channels = channels,
#             use_primary_edge_sampling=False,
#             use_secondary_edge_sampling=False,
#             sample_pixel_center = sample_pixel_center,
#             device = device)
#         imgs.append(pyredner.RenderFunction.apply(se, *scene_args))
#     imgs = torch.stack(imgs)
#     return imgs

# def renderPathTracing(scene,
#                       channels= None,
#                       max_bounces = 1,
#                       num_samples = 8,
#                       device = None):
#     if channels is None:
#         channels = [redner.channels.radiance]
#         channels.append(redner.channels.alpha)
#     #if alpha:
#     #   channels.append(redner.channels.alpha)
#     return rayTrace(scene=scene,
#                     channels=channels,
#                     max_bounces=max_bounces,
#                     sampler_type=pyredner.sampler_type.independent,
#                     num_samples=num_samples,
#                     seed = None,
#                     sample_pixel_center=False,
#                     device=device)

class RendererMitsuba:

    def __init__(self, samples, bounces, device):
        self.samples = samples
        self.bounces = bounces
        self.device = torch.device(device)
        self.near_clip = 10.0
        self.far_clip = 1000.0
        self.upVector = torch.tensor([0.0, -1.0, 0.0])
        self.counter = 0
        self.screenWidth = 256
        self.screenHeight = 256
        mi.set_variant('cuda_ad_rgb')
        


    def buildScenes(self, vertices, indices, normal, uv, diffuseTexture, specularTexture, roughnessTexture, focal, envMap):
        '''
        build multiple mitsuba scenes used for path tracing (uv mapping and indices are the same for all scenes)
        :param vertices: [n, verticesNumber, 3]
        :param indices: [indicesNumber, 3]
        :param normal: [n, verticesNumber, 3]
        :param uv: [verticesNumber, 2]
        :param diffuseTexture: [n, resX, resY, 3] or [1, resX, resY, 3]
        :param specularTexture: [n, resX, resY, 3] or [1, resX, resY, 3]
        :param roughnessTexture: [n, resX, resY, 1] or [1, resX, resY, 3]
        :param focal: [n]
        :param envMap: [n, resX, resY, 3]
        :return: return mitsuba scenes
        '''
        assert (vertices.dim() == 3 and vertices.shape[-1] == 3 and normal.dim() == 3 and normal.shape[-1] == 3)
        assert (indices.dim() == 2 and indices.shape[-1] == 3)
        assert (uv.dim() == 2 and uv.shape[-1] == 2)
        assert (diffuseTexture.dim() == 4 and diffuseTexture.shape[-1] == 3 and
                specularTexture.dim() == 4 and specularTexture.shape[-1] == 3 and
                roughnessTexture.dim() == 4 and roughnessTexture.shape[-1] == 1)
        assert(focal.dim() == 1)
        assert(envMap.dim() == 4 and envMap.shape[-1] == 3)
        assert(vertices.shape[0] == focal.shape[0] == envMap.shape[0])
        assert(diffuseTexture.shape[0] == specularTexture.shape[0] == roughnessTexture.shape[0])
        assert (diffuseTexture.shape[0] == 1 or diffuseTexture.shape[0] == vertices.shape[0])
        sharedTexture = True if diffuseTexture.shape[0] == 1 else False
       # GENERATE Textures
        self.diffuseTexture = diffuseTexture
        self.specularTexture = specularTexture
        self.roughnessTexture = roughnessTexture
       # GENERATE MESH
        # self.mesh = mi.Mesh(
        #     "my_mesh",
        #     vertex_count=vertices.shape[1],
        #     face_count=vertices.shape[1] - 1,
        #     has_vertex_normals=False,
        #     has_vertex_texcoords=False,
        # )
        # mesh_params = mi.traverse(self.mesh)
        # REVIEW does this slow down the code
        
        # vertices_np = vertices.squeeze(0).detach().cpu().numpy()
        # faces_np = indices.detach().cpu().numpy()
        # mesh_params["vertex_positions"] = dr.ravel(mi.Point3f(vertices_np))
        # mesh_params["vertex_positions"] = vertices
        # print('faces')
        # mesh_params["faces"] = indices.torch()
        # print('update')
        # print(mesh_params.update())
        # mesh_params.update()
        
        # https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/pytorch_mitsuba_interoperability.html
        
        return mi.load_dict({
            'type': 'scene',
            'integrator': {'type': 'prb'},
            'sensor':  {
                'type': 'perspective',
                'to_world': T.look_at(
                                origin=(0, 0, -2),
                                target=(0, 0, 0),
                                up=(0, -1, 0)
                            ),
                'fov': 60,
                'film': {
                    'type': 'hdrfilm',
                    'width':  self.screenWidth,
                    'height': self.screenHeight,
                },
            },
            'textured_plane': {
                'type': 'rectangle',
                'to_world': T.scale(1.2),
                'bsdf': {
                    'type': 'twosided',
                    'nested': {
                        'type': 'diffuse',
                        'reflectance': {
                            'type': 'bitmap',
                            'filename': "C:/Users/AQ14980/Desktop/repos/NextFace/output/Bikerman.jpg/diffuseMap_0.png"
                        },
                    }
                }
            },
            'glass_sphere': {
                'type': 'sphere',
                'to_world': T.translate([0, 0, -1]).scale(0.45),
                'bsdf': {
                    'type': 'dielectric',
                    'int_ior': 1.06,
                },
            },
            'light': {
                'type': 'constant',
            }
        })

    @dr.wrap_ad(source='torch', target='drjit')
    def render_texture(scene, spp=256, seed=1):
        """take a texture, update the scene and render it. uses a wrap ad for backpropagation and for gradients
        we are adding a mitsuba computations in a pytorch pipeline

        Args:
            texture (tensor): 
            scene (mitsuba scene ) : scene
            spp (int, optional): nb of samples. Defaults to 256.
            seed (int, optional): random ?. Defaults to 1.

        Returns:
            _type_: _description_
        """
        
        filename = "./output/Bikerman.jpg/diffuseMap_0.png"
        mi_texture =  mi.TensorXf(mi.Bitmap(filename).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32))
        params = mi.traverse(scene)
        key = 'textured_plane.bsdf.brdf_0.reflectance.data'
        params[key] = mi_texture
        params.update()
        image = mi.render(scene, params, spp=spp, seed=seed, seed_grad=seed+1)
        
        return image
        
        
    # def renderAlbedo(self, scenes):
    #     '''
    #     render albedo of given pyredner scenes
    #     :param scenes:  list of pyredner scenes
    #     :return: albedo images [n, screenWidth, screenHeight, 4]
    #     '''
    #     #images =pyredner.render_albedo(scenes, alpha = True, num_samples = self.samples, device = self.device)
    #     images = renderPathTracing(scenes,
    #                                channels= [pyredner.channels.diffuse_reflectance, pyredner.channels.alpha],
    #                                max_bounces = 0,
    #                                num_samples = self.samples ,
    #                                device = self.device)
    #     return images

    def render(self, scene):
        '''
        render scenes with ray tracing
        :param scene:  mitsuba scene
        :return: ray traced images [n, screenWidth, screenHeight, 4]
        '''
        self.counter += 1
        return RendererMitsuba.render_texture(scene)
    
#generate model
class Model1(torch.nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(256**2, 256**2),
            torch.nn.Sigmoid(),
        )

    def forward(self, texture):
        texture = texture.torch()
        # Evaluate the model one channel as a time
        rgb = [self.layers(texture[:, :, i].view(-1)) for i in range(3)]
        # Reconstruct and return the 3D tensor
        return torch.stack([c.view(256, 256) for c in rgb], dim=2)

