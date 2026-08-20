from pathlib import Path
import h5py, numpy as np, pandas as pd
from GWTC5_confirmatory_analysis import pair_metrics, find_model_group, XPHM, XPNR
ROOT=Path(__file__).resolve().parents[1]

def test_one_frozen_metric_matches_reference():
    event='GW240525_031210'; parameter='chirp_mass'
    path=ROOT/'data/processed/confirmatory_batch_A/confirmatory_batch_A_posteriors.hdf5'
    with h5py.File(path,'r') as h:
        eg=h[event]
        x=np.asarray(find_model_group(eg,XPHM)[parameter],dtype=float)
        y=np.asarray(find_model_group(eg,XPNR)[parameter],dtype=float)
    actual=pair_metrics(x[np.isfinite(x)],y[np.isfinite(y)])
    ref=pd.read_csv(ROOT/'reference_results/confirmatory_analysis/confirmatory_parameter_metrics.csv')
    row=ref[(ref.event==event)&(ref.parameter==parameter)].iloc[0]
    for key in ['wasserstein_1','normalized_wasserstein_1','js_divergence_bits','standardized_median_displacement','interval_overlap_90','log_width_ratio']:
        assert np.isclose(actual[key],row[key],rtol=1e-12,atol=1e-12)
