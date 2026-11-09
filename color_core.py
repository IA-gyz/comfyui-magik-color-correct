"""Independent NumPy color transfer implementations; float sRGB input/output."""
import numpy as np


def lab(rgb):
    a = np.asarray(rgb, dtype=np.float64)
    a = np.where(a <= .04045, a / 12.92, ((a + .055) / 1.055) ** 2.4)
    xyz = a @ np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]]).T
    xyz /= np.array([.95047, 1, 1.08883])
    d = 6/29
    f = np.where(xyz > d**3, np.cbrt(xyz), xyz/(3*d*d)+4/29)
    return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])], -1)


def rgb(l):
    y = (l[...,0]+16)/116
    f = np.stack([y+l[...,1]/500,y,y-l[...,2]/200],-1)
    d = 6/29
    xyz = np.where(f>d, f**3,3*d*d*(f-4/29))*[.95047,1,1.08883]
    a = xyz @ np.linalg.inv(np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]])).T
    return np.where(a<=.0031308,12.92*a,1.055*np.maximum(a,0)**(1/2.4)-.055)


def power(a, p):
    v,q = np.linalg.eigh(a)
    return (q*np.maximum(v,1e-6)**p)@q.T


def gaussian(x,s,r):
    ms,mr = s.mean(0),r.mean(0)
    cs = np.atleast_2d(np.cov(s,rowvar=False))+np.eye(3)*1e-5
    cr = np.atleast_2d(np.cov(r,rowvar=False))+np.eye(3)*1e-5
    root = power(cs,.5)
    inv = power(cs,-.5)
    t = inv@power(root@cr@root,.5)@inv
    return (x-ms)@t+mr


def quantile_map(x,s,r):
    # Collapse repeated source quantiles: constant channels map to reference median.
    q = np.linspace(0,1,min(1024,max(2,len(s),len(r))))
    a,b = np.quantile(s,q),np.quantile(r,q)
    u,idx,count = np.unique(a,return_index=True,return_counts=True)
    b = np.add.reduceat(b,idx)/count
    return np.interp(x,u,b)


def transfer(x,s,r,method,iterations=12,seed=0,check=lambda:None):
    if method == 'Reinhard (Lab)':
        ratio = r.std(0)/np.maximum(s.std(0),1e-3)
        return (x-s.mean(0))*np.minimum(ratio,20)+r.mean(0)
    if method == 'Gaussian optimal transport':
        return gaussian(x,s,r)
    if method == 'Histogram matching':
        return np.stack([quantile_map(x[:,i],s[:,i],r[:,i]) for i in range(3)],-1)
    if method == 'Histogram + Gaussian':
        y = transfer(x,s,r,'Histogram matching')
        ss = transfer(s,s,r,'Histogram matching')
        return gaussian(y,ss,r)
    if method != 'Sliced optimal transport':
        raise ValueError('Unknown color transfer method: '+method)
    rng = np.random.default_rng(seed)
    x,s = x.copy(),s.copy()
    for _ in range(iterations):
        check()
        basis,_ = np.linalg.qr(rng.normal(size=(3,3)))
        for d in basis.T:
            sp,rp = s@d,r@d
            delta = quantile_map(x@d,sp,rp)-x@d
            sd = quantile_map(sp,sp,rp)-sp
            x += .7*delta[:,None]*d
            s += .7*sd[:,None]*d
    return x


def smooth_spatial(x, radius, check=lambda: None):
    """Three separable box passes approximate Gaussian sigma=radius pixels.

    Edge extension avoids dark borders. Cumulative sums keep cost independent
    of radius and allow tiny/one-pixel dimensions.
    """
    half = max(1, int(round((np.sqrt(1 + 4*radius*radius)-1)/2)))
    out = np.asarray(x, dtype=np.float64)
    for _ in range(3):
        for axis in (0, 1):
            check()
            pad = [(0,0)] * out.ndim
            pad[axis] = (half,half)
            padded = np.pad(out,pad,mode='edge')
            pad[axis] = (1,0)
            cumulative = np.pad(np.cumsum(padded,axis=axis),pad,mode='constant')
            lo,hi = [slice(None)]*out.ndim,[slice(None)]*out.ndim
            lo[axis],hi[axis] = slice(None,-(2*half+1)),slice(2*half+1,None)
            out = (cumulative[tuple(hi)]-cumulative[tuple(lo)])/(2*half+1)
    return out


def upscale_restore(source_lab, reference_lab, radius_percent=2., color_strength=1., brightness_strength=1., check=lambda:None):
    """Add only smooth aligned Lab differences; source high frequencies remain.

    Reference must already be resized to source dimensions. Radius is approximate
    Gaussian sigma as percentage of the source short edge.
    """
    if source_lab.shape != reference_lab.shape:
        raise ValueError('Spatial restoration requires aligned images of equal size.')
    radius = max(.5,min(source_lab.shape[:2])*radius_percent/100.)
    delta = smooth_spatial(reference_lab-source_lab,radius,check)
    delta *= np.array([brightness_strength,color_strength,color_strength])
    return source_lab+delta
