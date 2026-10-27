"""Conservative train-only camera perturbations; complete glyphs remain visible."""
from PIL import Image,ImageDraw,ImageFilter


def camera_augment(face,rng,profile='none'):
    if profile=='none':return face
    if profile!='mild':raise ValueError('Unknown camera augmentation')
    side=face.width
    if rng.random()<.5:
        rotated=face.rotate(rng.uniform(-10,10),resample=Image.Resampling.BILINEAR,expand=True,fillcolor=(114,114,114))
        rotated.thumbnail((side,side),Image.Resampling.BILINEAR)
        face=Image.new('RGB',(side,side),(114,114,114));face.paste(rotated,((side-rotated.width)//2,(side-rotated.height)//2))
    if rng.random()<.15:
        # A translucent soft highlight, never an opaque erase/mask of glyphs.
        mask=Image.new('L',face.size,0);draw=ImageDraw.Draw(mask)
        x,y=rng.uniform(0,side),rng.uniform(0,side);radius=rng.uniform(.08,.18)*side
        draw.ellipse([x-radius,y-radius,x+radius,y+radius],fill=round(rng.uniform(.08,.22)*255))
        mask=mask.filter(ImageFilter.GaussianBlur(max(1,side*.025)))
        face=Image.composite(Image.new('RGB',face.size,'white'),face,mask)
    return face


class EnvelopeImage:
    """Expose the original global frame while storing only a lossless envelope."""
    def __init__(self,cached,rectangle,source_size):
        self.cached=cached;self.rectangle=rectangle
        self.width,self.height=source_size
        if cached.size!=(rectangle[2]-rectangle[0],rectangle[3]-rectangle[1]):
            raise ValueError('Cache pixels do not cover their signed global envelope')
    def crop(self,rectangle):
        l,t,r,b=rectangle;left,top,right,bottom=self.rectangle
        if not(left<=l<r<=right and top<=t<b<=bottom):
            raise ValueError('Requested global crop leaves the cached envelope')
        return self.cached.crop((l-left,t-top,r-left,b-top))


def cached_crop_face(cached,record,resolution,context,rng=None):
    from ml.inference.cascade_model import crop_face
    # All float arithmetic, jitter and integer rounding occur in the original
    # global frame; only already-integral crop coordinates are translated.
    wrapper=EnvelopeImage(cached,record['cacheEnvelopeXYXY'],record['sourceImageSize'])
    return crop_face(wrapper,record['bbox'],resolution,context,rng)
