from openslide import OpenSlide

name     = 'GTEX-1122O-1125'
slide    = OpenSlide('%s.svs' % name)
dims     = slide.dimensions
scale    = 100
new_dims = (dims[0]/100, dims[1]/100)
img      = slide.get_thumbnail(new_dims)

img.save('%s.png' % name)