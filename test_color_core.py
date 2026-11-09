"""Run directly; no torch or ComfyUI dependency required."""
import unittest
import numpy as np
from color_core import lab,rgb,transfer,upscale_restore,smooth_spatial

METHODS=['Gaussian optimal transport','Reinhard (Lab)','Histogram matching','Histogram + Gaussian','Sliced optimal transport']

class NumericalTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(4)
        self.a=rng.uniform(.1,.8,(512,3))
        self.x=lab(self.a)
        self.r=lab(rng.uniform(.2,.9,(600,3)))

    def test_roundtrip(self):
        np.testing.assert_allclose(rgb(self.x),self.a,atol=1e-6)

    def test_methods(self):
        for method in METHODS:
            with self.subTest(method=method):
                y=transfer(self.x,self.x,self.r,method)
                self.assertTrue(np.isfinite(y).all())
                np.testing.assert_array_equal(y,transfer(self.x,self.x,self.r,method))
                np.testing.assert_allclose(transfer(self.x,self.x,self.x,method),self.x,atol=1e-5)
                self.assertTrue(np.isfinite(transfer(np.zeros((4,3)),np.zeros((4,3)),self.r,method)).all())

    def test_gaussian_statistics(self):
        y=transfer(self.x,self.x,self.r,METHODS[0])
        np.testing.assert_allclose(y.mean(0),self.r.mean(0),atol=1e-6)
        np.testing.assert_allclose(np.cov(y,rowvar=False),np.cov(self.r,rowvar=False),atol=1e-3)

    def test_sliced_chunk_consistency(self):
        full=transfer(self.x,self.x,self.r,METHODS[-1],3)
        split=np.concatenate([transfer(a,self.x,self.r,METHODS[-1],3) for a in np.array_split(self.x,3)])
        np.testing.assert_allclose(full,split,atol=1e-9)

class SpatialTests(unittest.TestCase):
    def test_identity_and_bypass(self):
        a=np.random.default_rng(3).normal(size=(20,30,3))
        np.testing.assert_array_equal(upscale_restore(a,a),a)
        np.testing.assert_array_equal(upscale_restore(a,a+2,color_strength=0,brightness_strength=0),a)

    def test_known_cast_and_controls(self):
        a=np.full((24,32,3),[50.,10.,-4.])
        r=a+[5.,-3.,2.]
        np.testing.assert_allclose(upscale_restore(a,r),r,atol=1e-10)
        result=upscale_restore(a,r,color_strength=.5,brightness_strength=0)
        np.testing.assert_allclose(result,a+[0.,-1.5,1.],atol=1e-10)

    def test_detail_retention_and_locality(self):
        h,w=64,128
        ref=np.full((h,w,3),[50.,0.,0.])
        x=np.arange(w)
        detail=np.cos(x*np.pi)[None,:,None]*np.array([1.,.5,.5])
        source=ref+detail
        source[:,:w//2,1]+=8
        source[:,w//2:,1]-=8
        result=upscale_restore(source,ref,radius_percent=4)
        # Opposite local casts cannot be corrected by a single global offset.
        self.assertLess(abs(result[:,8:40,1].mean()),.1)
        self.assertLess(abs(result[:,88:120,1].mean()),.1)
        self.assertGreater(np.std(result[:,8:40,0]),.9)

    def test_tiny_dimensions(self):
        for shape in [(1,1,3),(1,8,3),(8,1,3)]:
            a=np.ones(shape)
            np.testing.assert_allclose(smooth_spatial(a,20),a)

if __name__=='__main__':
    unittest.main()
