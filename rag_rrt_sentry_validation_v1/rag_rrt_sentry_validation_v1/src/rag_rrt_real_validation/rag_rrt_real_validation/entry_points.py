"""Validate the generated Python executables before bringing up hardware."""
import ast
import os
from pathlib import Path
import sys
import tempfile


ENTRY_POINTS = {
    'rag_rrt_backend_py': ('planner_backend', 'rag_rrt_backend_py.backend_node'),
    'rag_rrt_real_validation': ('trial', 'rag_rrt_real_validation.runner'),
}


def verify_entry_points(prefixes, repair=False):
    results = []
    for package, (name, module) in ENTRY_POINTS.items():
        path = Path(prefixes[package])/'lib'/package/name
        if not path.parent.is_dir():
            raise RuntimeError('Package is not installed: '+str(path.parent))
        data = path.read_bytes() if path.is_file() else b''
        valid_header = data.startswith(b'#!') and b'\n' in data
        if not valid_header:
            if not repair:
                raise RuntimeError('Invalid or empty Python entry point: '+str(path)+
                                   '. Rebuild with bash scripts/build_workspace.sh.')
            text = '#!'+sys.executable+'\nfrom '+module+' import main\n\nif __name__ == "__main__":\n    main()\n'
            # Replace generated wrappers atomically, never truncate a live script.
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                stream.write(text.encode('utf-8'))
                temporary = Path(stream.name)
            temporary.chmod(0o755)
            os.replace(temporary, path)
            data = path.read_bytes()
        try:
            tree = ast.parse(data.decode('utf-8'), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            raise RuntimeError('Invalid Python syntax in '+str(path)) from exc
        if not tree.body:
            raise RuntimeError('Python entry point has no executable body: '+str(path))
        if not os.access(path, os.X_OK):
            if not repair:
                raise RuntimeError('Python entry point is not executable: '+str(path))
            path.chmod(path.stat().st_mode | 0o111)
        results.append(dict(package=package, executable=name, bytes=len(data), path=str(path)))
    return results
