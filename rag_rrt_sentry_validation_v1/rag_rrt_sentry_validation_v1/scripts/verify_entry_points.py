#!/usr/bin/env python3
"""Check and optionally repair colcon's generated console script wrappers."""
import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/rag_rrt_real_validation'))
from rag_rrt_real_validation.entry_points import ENTRY_POINTS, verify_entry_points

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repair', action='store_true')
parser.add_argument('--import-check', action='store_true')
args = parser.parse_args()
results = verify_entry_points({name: ROOT/'install'/name for name in ENTRY_POINTS}, repair=args.repair)
if args.import_check:
    for name, module in ENTRY_POINTS.values():
        if not callable(getattr(importlib.import_module(module), 'main', None)):
            raise RuntimeError('Missing main function: '+module)
print(json.dumps(dict(entry_points=results, imports_checked=args.import_check)))
