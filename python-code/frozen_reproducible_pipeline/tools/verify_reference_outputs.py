#!/usr/bin/env python3
from pathlib import Path
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]

def compare(actual,reference,keys,tol=1e-10):
    a=pd.read_csv(actual); r=pd.read_csv(reference)
    assert list(a.columns)==list(r.columns), f'column mismatch: {actual}'
    assert len(a)==len(r), f'row count mismatch: {actual}'
    for key in keys:
        av=pd.to_numeric(a[key],errors='coerce').to_numpy(float)
        rv=pd.to_numeric(r[key],errors='coerce').to_numpy(float)
        if not np.allclose(av,rv,rtol=tol,atol=tol,equal_nan=True):
            raise AssertionError(f'numeric mismatch {key}: {actual}')

compare(ROOT/'results/confirmatory_analysis/confirmatory_hypothesis_results.csv',ROOT/'reference_results/confirmatory_analysis/confirmatory_hypothesis_results.csv',['effect_statistic','raw_p_value','adjusted_p_value'])
compare(ROOT/'results/confirmatory_analysis/confirmatory_event_endpoints.csv',ROOT/'reference_results/confirmatory_analysis/confirmatory_event_endpoints.csv',['event_median_NW1','maximum_screening_shift','screening_max_NW1'])
compare(ROOT/'results/robustness_analysis/robustness_leave_one_out_summary.csv',ROOT/'reference_results/robustness_analysis/robustness_leave_one_out_summary.csv',['effect_min','effect_max','p_value_min','p_value_max'])
compare(ROOT/'results/post_confirmatory_sensitivity/design_aware_sensitivity.csv',ROOT/'reference_results/post_confirmatory_sensitivity/design_aware_sensitivity.csv',['effect','p_value','lower_95','upper_95'])
compare(ROOT/'results/post_confirmatory_sensitivity/h3_parameterization_summary.csv',ROOT/'reference_results/post_confirmatory_sensitivity/h3_parameterization_summary.csv',['median_first','median_second','median_difference','wilcoxon_W','p_value'])
print('All key reproduced tables match the archived reference outputs within tolerance.')
