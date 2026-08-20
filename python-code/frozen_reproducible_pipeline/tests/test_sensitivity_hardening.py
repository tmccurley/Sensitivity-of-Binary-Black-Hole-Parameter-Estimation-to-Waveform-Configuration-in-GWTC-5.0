from pathlib import Path
import h5py, numpy as np, pytest
from paper_sensitivity_revision import (
    XPHM, XPNR, locate_posteriors, load, within_stratum_permutation
)

def make_file(path: Path, event: str, values=None):
    values=np.asarray(values if values is not None else [1.,2.,3.,4.])
    with h5py.File(path,'w') as h:
        eg=h.create_group(event)
        for i,label in enumerate([XPHM,XPNR]):
            g=eg.create_group(f'model{i}')
            g.attrs['original_label']=label.encode()
            g.create_dataset('chi_eff',data=values)

def test_event_file_collision_is_rejected(tmp_path):
    make_file(tmp_path/'a_posteriors.hdf5','GWTEST_000000')
    make_file(tmp_path/'b_posteriors.hdf5','GWTEST_000000')
    with pytest.raises(RuntimeError,match='multiple compact files'):
        locate_posteriors(tmp_path,{'GWTEST_000000'})

def test_nonfinite_values_are_removed(tmp_path):
    make_file(tmp_path/'a_posteriors.hdf5','GWTEST_000001',[1.,2.,np.nan,np.inf,3.,4.])
    mapping=locate_posteriors(tmp_path,{'GWTEST_000001'})
    values=load(mapping,'GWTEST_000001',XPHM,'chi_eff')
    assert np.array_equal(values,np.array([1.,2.,3.,4.]))

def test_within_stratum_permutation_is_deterministic():
    x=np.arange(9.); y=np.array([0,2,1,3,5,4,6,8,7.]); s=np.repeat(['a','b','c'],3)
    a=within_stratum_permutation(x,y,s,seed=42,permutations=1000)
    b=within_stratum_permutation(x,y,s,seed=42,permutations=1000)
    assert a==b
