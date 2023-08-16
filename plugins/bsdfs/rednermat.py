import numpy as np
import mitsuba as mi
import drjit as dr
import math
from typing import Tuple

# Redner material implementation inside Mitsuba
# We are assuming all component used
class RednerMat(mi.BSDF):
    def __init__(self, props: mi.Properties) -> None:
        super().__init__(props)
        
        self.albedo: mi.Texture = props.get("albedo", mi.Texture.D65(0.1))
        self.roughness: mi.Texture2f = props.get("roughness", mi.Texture2f(0.1)) # TODO: Check this type
        self.specular: mi.Texture = props.get("specular", mi.Texture.D65(0.1))
    
    def roughness_to_phong(roughness: mi.Float) -> mi.Float:
        return dr.max(2.0 / roughness - 2, 0)
    
    def smithG1(roughness: mi.Float, v: mi.Vector3f) -> mi.Float:
        cos_theta = v.z
        # tan^2 + 1 = 1/cos^2
        tan_theta = dr.safe_sqrt(dr.max(1 / (cos_theta * cos_theta) - 1.0, 0))
        
        alpha = dr.sqrt(roughness)
        a = 1.0 / (alpha * tan_theta)
        result = dr.select(
            a > 1.6, 1, 
            (3.535 * a + 2.181 * a * a) / (1.0 + 2.276 * a + 2.577 * a*a)
        )
        result[tan_theta == 0] = 1
        return result
        
    
    def eval(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f, wo: mi.Vector3f, active: bool = True) -> mi.Color3f:
        # Calcul contribution diffuse
        diffuse_contrib = self.albedo(si, active) * wo.z / mi.Float(math.pi)
        
        # Calcul contribution speculaire        
        m = dr.normalize(si.wi + wo)
        roughness = dr.max(self.roughness.eval(si.uv),0.00001)
        phong_exponent = self.roughness_to_phong(roughness)
        specular_reflectance = self.specular.eval(si, active)
        
        D = dr.power(m.z, phong_exponent) * (phong_exponent + 2.0) / mi.Float(2 * math.pi)
        G = self.smithG1(si.wi) * self.smithG1(wo)
        F = specular_reflectance + (1 - specular_reflectance) * dr.power(dr.max(1 - dr.abs(dr.dot(m, wo)), 0), 5)
        specular_contrib = dr.select(wo.z > 0, F * D * G / (4.0 * wo.z), 0)
        
        return specular_contrib + diffuse_contrib
        
    
    def pdf(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f, wo: mi.Vector3f, active: bool = True) -> float:
        # Compute the probability of selecting the diffuse and specular lobe
        diffuse_pmf = mi.luminance(self.albedo.eval(si, active))
        specular_pmf = mi.luminance(self.specular.eval(si, active))
        weight_pmf = diffuse_pmf + specular_pmf
        diffuse_pmf = dr.select(weight_pmf > 0.0, diffuse_pmf / weight_pmf, 0.0)
        specular_pmf = dr.select(weight_pmf > 0.0, specular_pmf / weight_pmf, 0.0)
        
        # Compute diffuse PDF
        diffuse_pdf = dr.select(diffuse_pmf > 0, diffuse_pmf * wo.z / math.pi, 0.0)
        
        # Compute specular PDF
        m = dr.normalize(si.wi + wo)
        roughness = dr.max(self.roughness.eval(si.uv),0.00001)
        phong_exponent = self.roughness_to_phong(roughness)
        D = dr.power(m.z, phong_exponent) * (phong_exponent + 2.0) / mi.Float(2 * math.pi)
        specular_pdf = dr.select(dr.abs(m, wo) > 0,  specular_pmf * D * m.z / (4.0 * dr.abs(dr.dot(m, wo))), 0.0)
        
        return diffuse_pdf + specular_pdf
    
    def eval_pdf(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f, wo: mi.Vector3f, active: bool = True) -> Tuple[mi.Color3f, float]:
        # TODO: Could be more efficient by fusing some computations
        return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)
    
    def sample(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f, sample1: float, sample2: mi.Point2f, active: bool = True) -> Tuple[mi.BSDFSample3f, mi.Color3f]:
        # Compute PDF selection
        diffuse_pmf = mi.luminance(self.albedo.eval(si, active))
        specular_pmf = mi.luminance(self.specular.eval(si, active))
        weight_pmf = diffuse_pmf + specular_pmf
        diffuse_pmf = dr.select(weight_pmf > 0.0, diffuse_pmf / weight_pmf, 0.0)
        specular_pmf = dr.select(weight_pmf > 0.0, specular_pmf / weight_pmf, 0.0)
        
        # Example to do specular and diffuse sampling
        # https://github.com/mitsuba-renderer/mitsuba3/blob/6105dfc1057a3bc204393b8e9557bcb26e08568b/src/bsdfs/pplastic.cpp#L243C9-L256C10
        # Uses a lot of masks
        
        # TODO: Sample diffuse
        # TODO: Sample specular
        
        # Ressources: 
        # - https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html (official doc for BSDF)
        # - https://github.com/rgl-epfl/differentiable-sdf-rendering/blob/main/python/integrators/sdf_direct_reparam.py (project that use a lot of mitsuba)
        
        pass
    
    def to_string(self):
        return ('RednerMat[\n'
                '    albedo=%s,\n'
                '    roughness=%s,\n'
                '    specular=%s,\n'
                ']' % (self.albedo, self.roughness, self.specular))
    
    def traverse(self, callback):
        callback.put_parameter('albedo', self.albedo, mi.ParamFlags.Differentiable)
        callback.put_parameter('roughness', self.roughness, mi.ParamFlags.Differentiable)
        callback.put_parameter('specular', self.specular, mi.ParamFlags.Differentiable)