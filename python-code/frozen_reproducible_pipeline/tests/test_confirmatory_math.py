import numpy as np
from GWTC5_confirmatory_analysis import (
    pair_metrics, js_divergence_bits, holm_adjust, spearman_permutation_test
)

def test_identical_distributions():
    x=np.linspace(-2,2,401)
    m=pair_metrics(x,x.copy())
    assert m['wasserstein_1']==0.0
    assert m['normalized_wasserstein_1']==0.0
    assert m['js_divergence_bits']==0.0
    assert m['interval_overlap_90']==1.0
    assert abs(m['log_width_ratio'])<1e-15

def test_shifted_distribution_wasserstein():
    x=np.linspace(-1,1,501); y=x+0.5
    m=pair_metrics(x,y)
    assert np.isclose(m['wasserstein_1'],0.5,atol=1e-12)
    assert m['normalized_wasserstein_1']>0
    assert m['interval_overlap_90']<1

def test_jsd_symmetry_and_nonnegative():
    x=np.linspace(0,1,500); y=np.linspace(0.3,1.3,500)
    a=js_divergence_bits(x,y); b=js_divergence_bits(y,x)
    assert a>=0
    assert np.isclose(a,b,atol=1e-15)

def test_disjoint_interval_overlap_zero():
    x=np.linspace(0,1,1001); y=np.linspace(10,11,1001)
    assert pair_metrics(x,y)['interval_overlap_90']==0.0

def test_holm_adjustment():
    out=holm_adjust({'H2':9.99990000099999e-06,'H3':3.814697265625e-06})
    assert np.isclose(out['H3'],7.62939453125e-06)
    assert np.isclose(out['H2'],9.99990000099999e-06)

def test_permutation_is_seed_deterministic():
    x=np.arange(8.0); y=np.array([0,1,3,2,5,4,7,6.],float)
    a=spearman_permutation_test(x,y,np.random.default_rng(123),n_permutations=1000)
    b=spearman_permutation_test(x,y,np.random.default_rng(123),n_permutations=1000)
    assert a==b
