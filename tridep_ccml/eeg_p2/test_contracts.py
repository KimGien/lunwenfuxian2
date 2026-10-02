from __future__ import annotations
import unittest
from .common import *
from .quality import structure, spectral, measures
from .scales import fit,subject_transform,METHODS


class Contracts(unittest.TestCase):
    def test_reference_is_not_bad_channel(self):
        x=np.vstack([np.sin(np.arange(1000)/3),np.zeros(1000)])
        out=structure(x,['E1','Cz'],250)
        self.assertFalse(out['channels'][1]['flat_review'])
        self.assertFalse(out['channels'][1]['extreme_plateau_review'])

    def test_flat_plateau_and_duplicates(self):
        y=np.sin(np.arange(1000)/7)
        y[100:400]=4.
        out=structure(np.vstack([y,y]),['E1','E2'],250)
        self.assertTrue(out['channels'][0]['flat_review'])
        self.assertTrue(out['channels'][0]['extreme_plateau_review'])
        self.assertEqual(out['exact_duplicate_active_channels'],[['E1','E2']])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            structure(np.array([[0.,np.nan]]),['E1'],250)

    def test_frequency_band_and_line_diagnostics(self):
        t=np.arange(2000)/200
        x=np.array([np.sin(2*np.pi*f*t) for f in [2,10,50,60]])
        r=spectral(x,200)
        self.assertGreater(r['low_fraction'][0],.99)
        self.assertLess(r['low_fraction'][1],1e-8)
        self.assertGreater(r['line_ratio'][2],10)
        self.assertGreater(r['high_fraction'][3],.99)

    def test_shared_fit_ignores_nontrain_values(self):
        stats={s:{'mean':[i,2*i],'std':[i+1,i+2],'median':[i,2*i],'mad_sigma':[i+1,i+2]} for i,s in enumerate(['a','b','test'])}
        for method in METHODS[:2]:
            before=fit(stats,['a','b'],['a','b'],method)
            stats['test']={k:[1e10,1e10] for k in stats['test']}
            after=fit(stats,['a','b'],['a','b'],method)
            for a,b in zip(before,after):
                np.testing.assert_array_equal(a,b)
            with self.assertRaises(ValueError):
                fit(stats,['a','test'],['a','b'],method)

    def test_subject_transform_is_local_and_affine(self):
        x=np.random.default_rng(0).normal(size=(3,1000))*np.array([1,5,10])[:,None]+4
        for method in METHODS[2:]:
            z,_,_=subject_transform(x,method)
            changed=x*7+200
            z2,_,_=subject_transform(changed,method)
            np.testing.assert_allclose(z,z2,rtol=1e-12,atol=1e-12)
            with self.assertRaises(ValueError):
                subject_transform(np.zeros_like(x),method)


if __name__=='__main__':
    unittest.main()
