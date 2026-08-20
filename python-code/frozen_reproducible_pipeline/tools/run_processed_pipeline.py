#!/usr/bin/env python3
"""Reproduce the processed-data analysis stages using relative paths."""
from pathlib import Path
import argparse, shutil, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]

def run(cmd):
    print('+',' '.join(map(str,cmd)),flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--overwrite',action='store_true'); p.add_argument('--skip-tests',action='store_true'); a=p.parse_args()
    if not a.skip_tests: run([sys.executable,'-m','pytest','-q'])
    for rel in ['results/confirmatory_analysis','results/robustness_analysis','results/post_confirmatory_sensitivity']:
        path=ROOT/rel
        if path.exists() and any(path.iterdir()):
            if not a.overwrite: raise SystemExit(f'{path} is not empty; use --overwrite for a documented rerun')
            shutil.rmtree(path)
    run([sys.executable,str(ROOT/'03_confirmatory_analysis/analysis/GWTC5_confirmatory_analysis.py'),'--project-root',str(ROOT)])
    run([sys.executable,str(ROOT/'04_post_confirmatory_robustness/analysis/GWTC5_robustness_analysis.py'),'--project-root',str(ROOT)])
    run([sys.executable,str(ROOT/'04_post_confirmatory_robustness/analysis/paper_sensitivity_revision.py'),'--project-root',str(ROOT),'--output-dir',str(ROOT/'results/post_confirmatory_sensitivity')])
    run([sys.executable,str(ROOT/'tools/verify_reference_outputs.py')])
if __name__=='__main__': main()
