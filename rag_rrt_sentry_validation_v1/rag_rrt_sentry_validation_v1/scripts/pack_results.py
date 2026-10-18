#!/usr/bin/env python3
import argparse
from pathlib import Path
import tarfile
import tempfile

parser=argparse.ArgumentParser(description='Pack explicit result folders without deleting them')
parser.add_argument('folders',nargs='+')
parser.add_argument('--output-dir',default=str(Path.home()))
args=parser.parse_args()
paths=[Path(p).expanduser().resolve() for p in args.folders]
for p in paths:
    if not p.is_dir():parser.error('result directory missing: '+str(p))
output_dir=Path(args.output_dir).expanduser().resolve();output_dir.mkdir(parents=True,exist_ok=True)
file=tempfile.NamedTemporaryFile(prefix='rag_rrt_real_results_',suffix='.tar.gz',dir=output_dir,delete=False);file.close()
target=Path(file.name)
with tarfile.open(target,'w:gz') as tar:
    for index,p in enumerate(paths):
        # Never include the output archive when output is inside a selected folder.
        def keep(info):
            original=p / Path(info.name).relative_to(f'{index}_{p.name}')
            return None if original.resolve()==target else info
        tar.add(p,arcname=f'{index}_{p.name}',filter=keep)
print(target)
