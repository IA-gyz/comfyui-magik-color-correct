import numpy as np
import torch
import torch.nn.functional as F
import comfy.model_management as mm
from comfy.utils import ProgressBar
from .color_core import lab, rgb, transfer, upscale_restore

METHODS = ['Gaussian optimal transport','Reinhard (Lab)','Histogram matching','Histogram + Gaussian','Sliced optimal transport','Upscale color restore']


def validate_image(x,name):
    if x.ndim != 4 or x.shape[-1] not in (3,4) or min(x.shape[:3]) < 1:
        raise ValueError(f'{name} must be a nonempty BHWC RGB/RGBA IMAGE.')
    if not torch.isfinite(x).all():
        raise ValueError(f'{name} contains NaN or infinity.')


def mask_array(mask,index,count,h,w,name):
    if mask is None:
        return None
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or mask.shape[0] not in (1,count):
        raise ValueError(f'{name}: use one mask or one per corresponding image.')
    m = mask[0 if mask.shape[0]==1 else index].detach().float().cpu()
    if not torch.isfinite(m).all():
        raise ValueError(f'{name} contains NaN or infinity.')
    return F.interpolate(m[None,None],size=(h,w),mode='bilinear',align_corners=False)[0,0].clamp(0,1).numpy()


def samples(x,mask,limit,seed):
    p = x.reshape(-1,3)
    if mask is not None:
        p = p[mask.reshape(-1)>.5]
    if len(p)<2:
        raise ValueError('Statistics mask must select at least two pixels (values above 0.5).')
    if len(p)>limit:
        p = p[np.random.default_rng(seed).choice(len(p),limit,replace=False)]
    return lab(p)


class MagikColorCorrect:
    CATEGORY = 'Magik/Image'
    FUNCTION = 'correct'
    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('image',)
    DESCRIPTION = 'Transfer reference color statistics to an image. Different sizes are supported. Upscale color restore requires matching composition; other methods do not require alignment. White masks select pixels. See tooltips for method and batch behavior.'

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'image': ('IMAGE',{'tooltip':'Image or batch to recolor; resolution and alpha are preserved.'}),
            'reference': ('IMAGE',{'tooltip':'Image whose color distribution you want to copy.'}),
            'method': (METHODS,{'default':METHODS[0],'tooltip':'Gaussian: balanced covariance transfer. Reinhard: fast mean/std in Lab. Histogram: channel distributions. Hybrid: histogram then Gaussian. Sliced OT: iterative distribution matching; slower. Upscale color restore: spatial correction for aligned original/upscaled pairs.'}),
            'strength': ('FLOAT',{'default':1.,'min':0.,'max':2.,'step':.01,'tooltip':'0 = original, 1 = full correction, above 1 exaggerates the correction.'}),
            'preserve_luminance': ('FLOAT',{'default':0.,'min':0.,'max':1.,'step':.01,'tooltip':'0 copies reference brightness; 1 restores original CIELAB lightness. Gamut clipping can still alter final lightness.'}),
            'reference_mode': (['First reference','Pair by index'],{'tooltip':'First reuses reference 1 for all images. Pair requires one reference or exactly the image batch size.'}),
            'sample_pixels': ('INT',{'default':65536,'min':1024,'max':1048576,'step':1024,'tooltip':'Maximum pixels per image used to estimate distributions. Output remains full resolution.'}),
            'iterations': ('INT',{'default':12,'min':1,'max':64,'tooltip':'Sliced optimal transport only: more passes improve distribution matching but take longer.'}),
            'seed': ('INT',{'default':0,'min':0,'max':2147483647,'tooltip':'Deterministic pixel sampling and sliced transport directions.'}),
        },'optional':{
            'color_strength': ('FLOAT',{'default':1.,'min':0.,'max':2.,'step':.01,'tooltip':'Upscale color restore only: restore local Lab chroma. 0 leaves chroma unchanged; 1 restores reference low-frequency colors.'}),
            'brightness_strength': ('FLOAT',{'default':1.,'min':0.,'max':2.,'step':.01,'tooltip':'Upscale color restore only: restore local Lab lightness. Multiplied by (1 - preserve_luminance).'}),
            'correction_radius': ('FLOAT',{'default':2.,'min':.1,'max':20.,'step':.1,'tooltip':'Upscale color restore only: approximate blur sigma as percent of output short edge. 2% = about 41 pixels at 2048px. Smaller follows local colors; larger tolerates small structural differences.'}),
            'mask': ('MASK',{'tooltip':'Where to apply correction: white = corrected, black = original. Soft edges blend.'}),
            'source_statistics_mask': ('MASK',{'tooltip':'Select source colors used to learn the transform (>0.5). Does not restrict where correction is applied.'}),
            'reference_mask': ('MASK',{'tooltip':'Select reference colors used to learn the transform (>0.5). Useful for excluding backgrounds.'}),
        }}

    def correct(self,image,reference,method=METHODS[0],strength=1.,preserve_luminance=0.,reference_mode='First reference',sample_pixels=65536,iterations=12,seed=0,mask=None,source_statistics_mask=None,reference_mask=None,color_strength=1.,brightness_strength=1.,correction_radius=2.):
        validate_image(image,'image')
        validate_image(reference,'reference')
        n,nr = len(image),len(reference)
        if reference_mode not in ('First reference','Pair by index'):
            raise ValueError('Unknown reference_mode.')
        if reference_mode=='Pair by index' and nr not in (1,n):
            raise ValueError('Pair by index requires one reference or exactly one per source image.')
        if method not in METHODS:
            raise ValueError('Unknown method.')
        if not (np.isfinite(strength) and 0<=strength<=2 and 0<=preserve_luminance<=1):
            raise ValueError('Invalid strength or luminance preservation.')
        if not (1024<=sample_pixels<=1048576 and 1<=iterations<=64 and 0<=seed<=2147483647):
            raise ValueError('Invalid sampling, iterations or seed.')
        spatial = method=='Upscale color restore'
        if spatial and not (0<=color_strength<=2 and 0<=brightness_strength<=2 and .1<=correction_radius<=20):
            raise ValueError('Invalid upscale restoration controls.')
        if spatial and (source_statistics_mask is not None or reference_mask is not None):
            raise ValueError('Upscale color restore uses corresponding locations, not statistics masks. Disconnect source_statistics_mask/reference_mask; use mask to restrict correction.')
        if spatial and abs((image.shape[2]/image.shape[1])/(reference.shape[2]/reference.shape[1])-1) > .01:
            raise ValueError('Upscale color restore requires the same framing and aspect ratio (within 1%). Use an aligned reference or another method.')
        if strength==0 or (spatial and color_strength==0 and (brightness_strength==0 or preserve_luminance==1)):
            return (image,)
        result = torch.empty_like(image)
        progress = ProgressBar(n)
        refs = {}
        for i in range(n):
            mm.throw_exception_if_processing_interrupted()
            j = i if reference_mode=='Pair by index' and nr>1 else 0
            src = image[i,...,:3].detach().float().cpu().numpy().clip(0,1)
            h,w = src.shape[:2]
            sm = mask_array(source_statistics_mask,i,n,h,w,'source_statistics_mask')
            apply = mask_array(mask,i,n,h,w,'mask')
            if spatial:
                # Resize sRGB reference with antialiasing when downsampling.
                ref_tensor = reference[j,...,:3].detach().float().cpu().permute(2,0,1)[None]
                ref = F.interpolate(ref_tensor,size=(h,w),mode='bilinear',align_corners=False,antialias=True)[0].permute(1,2,0).numpy().clip(0,1)
                original = lab(src)
                out = upscale_restore(original,lab(ref),correction_radius,color_strength,
                                      brightness_strength*(1-preserve_luminance),
                                      mm.throw_exception_if_processing_interrupted)
            else:
                if j not in refs:
                    ref = reference[j,...,:3].detach().float().cpu().numpy().clip(0,1)
                    rm = mask_array(reference_mask,j,nr,*ref.shape[:2],'reference_mask')
                    refs[j] = samples(ref,rm,sample_pixels,seed)
                s = samples(src,sm,sample_pixels,seed)
                original = lab(src).reshape(-1,3)
                # Chunk full-resolution transforms to bound temporary memory, even for iterative OT.
                out = np.empty_like(original)
                for start in range(0,len(original),131072):
                    mm.throw_exception_if_processing_interrupted()
                    end = start+131072
                    t = transfer(original[start:end],s,refs[j],method,iterations,seed,mm.throw_exception_if_processing_interrupted)
                    t[:,0] = t[:,0]*(1-preserve_luminance)+original[start:end,0]*preserve_luminance
                    out[start:end] = t
            corrected = np.clip(rgb(out.reshape(h,w,3)),0,1)
            alpha = strength if apply is None else strength*apply[...,None]
            mixed = np.clip(src+alpha*(corrected-src),0,1)
            if not np.isfinite(mixed).all():
                raise ValueError(f'Color transfer produced nonfinite pixels in frame {i+1}.')
            result[i,...,:3] = torch.from_numpy(mixed.astype(np.float32)).to(device=image.device,dtype=image.dtype)
            if image.shape[-1]==4:
                result[i,...,3] = image[i,...,3]
            progress.update(1)
        return (result,)
