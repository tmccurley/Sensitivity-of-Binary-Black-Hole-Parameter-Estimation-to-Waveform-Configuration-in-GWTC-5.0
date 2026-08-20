#!/usr/bin/env python3
"""Verify that all public GWTC-5 inputs needed for source preparation exist."""
from pathlib import Path
import argparse, pandas as pd

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args(); root=args.project_root.resolve()
    manifest=pd.read_csv(root/'01_public_source_preparation/manifests/public_event_files.csv')
    event_dir=root/'data/raw/events'
    missing=[name for name in manifest.expected_filename if not (event_dir/name).exists()]
    catalog=list((root/'data/raw/catalog').glob('*PESummaryTable*.hdf5'))
    if len(catalog)!=1: print(f'Expected one catalog summary HDF5; found {len(catalog)}')
    if missing:
        print(f'Missing {len(missing)} public event files:')
        for name in missing: print('  ',name)
        raise SystemExit(1)
    print(f'All {len(manifest)} event files and one catalog summary file are present.')
if __name__=='__main__': main()
