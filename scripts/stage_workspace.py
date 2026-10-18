#!/usr/bin/env python3
"""Stage complete source snapshots, including files inside nested repositories.

Never moves/deletes nested .git metadata. Uses Git's normal clean filters (LFS,
line endings) and index format, avoiding gitlinks. Run at the workspace root.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'.git', 'build', 'install', 'log', '__pycache__', '.pytest_cache', '.cache', '.vscode'}
ROOT_SKIP = {'results', 'sentry_maps', 'work'}


def run(args, data=None):
    return subprocess.run(['git', *args], cwd=ROOT, input=data, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True).stdout


def files():
    for directory, dirs, names in os.walk(ROOT):
        base = Path(directory)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not (base == ROOT and d in ROOT_SKIP))
        for name in sorted(names):
            if base == ROOT and name == 'src.zip':
                continue
            if (name == '.git' or name.endswith(('.pyc', '.pem', '.key')) or
                (name.startswith('.env') and name != '.env.example')):
                continue
            p = base/name
            if p.is_symlink():
                raise RuntimeError('Review symlink before snapshot upload: '+str(p))
            yield p.relative_to(ROOT).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='inventory only; do not stage')
    args = parser.parse_args()
    if Path(os.fsdecode(run(['rev-parse', '--show-toplevel'])).strip()).resolve() != ROOT:
        raise RuntimeError('This script must belong to the top-level workspace repository')
    paths = sorted(files())
    nested = sorted(str(p.parent.relative_to(ROOT)) for p in ROOT.rglob('.git')
                    if p.parent != ROOT and not any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts[:-1]))
    if args.check:
        print(json.dumps(dict(files=len(paths), bytes=sum((ROOT/p).stat().st_size for p in paths),
                              nested_repositories=nested), ensure_ascii=True, indent=2))
        return
    run(['lfs', 'version'])
    # C-quoted UTF-8 path names allow spaces and non-ASCII characters.
    input_paths = ''.join(json.dumps(p, ensure_ascii=False)+'\n' for p in paths).encode('utf-8')
    print('Applying Git/LFS filters to {} files...'.format(len(paths)), flush=True)
    hashes = run(['hash-object', '-w', '--stdin-paths'], input_paths).decode().splitlines()
    if len(hashes) != len(paths):
        raise RuntimeError('Unexpected number of Git object hashes')
    previous = set(os.fsdecode(p) for p in run(['ls-files', '-z']).split(b'\0') if p)
    records = []
    for missing in sorted(previous-set(paths)):
        records.append(('0 '+('0'*40)+'\t'+missing).encode('utf-8')+b'\0')
    for path, oid in zip(paths, hashes):
        mode = '100755' if path.endswith('.sh') else '100644'
        records.append((mode+' '+oid+'\t'+path).encode('utf-8')+b'\0')
    run(['update-index', '-z', '--index-info'], b''.join(records))
    staged = run(['ls-files', '--stage']).decode('utf-8', errors='replace').splitlines()
    if any(line.startswith('160000 ') for line in staged):
        raise RuntimeError('Unexpected gitlink remains in the workspace index')
    print('Staged {} files as full snapshots; {} nested repositories preserved locally.'.format(len(paths), len(nested)))
    print('Review: git diff --cached --stat; then git commit and git push.')


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr.decode('utf-8', errors='replace'))
        raise SystemExit(exc.returncode)
