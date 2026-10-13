"""Fail closed unless called inside a fresh, marked reproduction workspace."""
import os
from pathlib import Path

def work_root():
    value=os.environ.get('MPN_RUN_ROOT')
    if not value:
        raise RuntimeError('Use run_pipeline.py; MPN_RUN_ROOT is required.')
    root=Path(value).resolve()
    if not (root/'.mpn_reproduction_run').is_file():
        raise RuntimeError('Refusing an unmarked output directory.')
    return root
