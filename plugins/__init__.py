import mitsuba as mi

from plugins.bsdfs.rednermat import RednerMat

# BSDFs
mi.register_bsdf("rednermat", lambda props: RednerMat(props))