import mitsuba as mi

from plugins.bsdfs.rednermat import RednerMat
# from plugins.bsdfs.mitsubaBSDF import MitsubaBSDF
# BSDFs
mi.register_bsdf("rednermat", lambda props: RednerMat(props))
# mi.register_bsdf("mitsubaBSDF", lambda props: MitsubaBSDF(props))

# Control variate
from .twostatebsdf import TwoStateBSDF
from .twostatemedium import TwoStateMedium
from .twostatepath import *
from .twostatevolpath import *
from .metaintegrator import MetaIntegrator
from .volpathsimple import VolpathSimpleIntegrator
from .cv_integrator import CVIntegrator

mi.register_integrator("volpathsimple", lambda props: VolpathSimpleIntegrator(props))
mi.register_integrator("twostateprb", TwoStatePRBIntegrator)
mi.register_integrator("meta", MetaIntegrator)
mi.register_integrator("cv", CVIntegrator)
mi.register_integrator("twostateprbvolpath", TwoStatePRBVolpathIntegrator)
mi.register_integrator("twostateprbpath", TwoStatePRBIntegrator)
mi.register_bsdf('twostate', TwoStateBSDF)
mi.register_medium('twostatemedium', TwoStateMedium)