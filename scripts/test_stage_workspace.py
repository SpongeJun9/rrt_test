"""Exercise snapshot staging in an isolated temporary Git repository."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('stage_workspace', Path(__file__).with_name('stage_workspace.py'))
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        subprocess.run(['git', '-C', str(self.root), 'lfs', 'install', '--local'], check=True, capture_output=True)
        self.original_root = stage.ROOT
        stage.ROOT = self.root
        self.addCleanup(setattr, stage, 'ROOT', self.original_root)
        (self.root/'.gitattributes').write_text('*.pcd filter=lfs diff=lfs merge=lfs -text\n', encoding='utf-8')
        self.nested = self.root/'src/library'
        self.nested.mkdir(parents=True)
        subprocess.run(['git', 'init', '-q', str(self.nested)], check=True)
        (self.nested/'.gitignore').write_text('*.pcd\n', encoding='utf-8')
        (self.nested/'package.xml').write_text('<package/>\n', encoding='utf-8')
        (self.nested/'地图 空间.pcd').write_bytes(b'example point cloud\n')

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args])

    def stage(self):
        with patch.object(sys, 'argv', ['stage_workspace.py']), contextlib.redirect_stdout(io.StringIO()):
            stage.main()

    def test_nested_source_and_lfs_filter_without_gitlinks(self):
        metadata = (self.nested/'.git/config').read_bytes()
        self.stage()
        index = self.git('ls-files', '--stage')
        self.assertIn(b'src/library/package.xml', index)
        self.assertNotIn(b'160000 ', index)
        self.assertEqual((self.nested/'.git/config').read_bytes(), metadata)
        pointer = self.git('show', ':src/library/地图 空间.pcd')
        self.assertTrue(pointer.startswith(b'version https://git-lfs.github.com/spec/v1\n'))

    def test_deleted_source_removed_from_index(self):
        self.stage()
        (self.nested/'package.xml').unlink()
        self.stage()
        self.assertNotIn(b'src/library/package.xml', self.git('ls-files'))

    def test_local_artifacts_and_metadata_excluded(self):
        (self.root/'src.zip').write_bytes(b'local archive')
        (self.root/'.env').write_text('LOCAL=only\n', encoding='utf-8')
        (self.root/'build').mkdir()
        (self.root/'build/output').write_text('generated', encoding='utf-8')
        self.stage()
        paths = self.git('ls-files', '-z').split(b'\0')
        self.assertNotIn(b'src.zip', paths)
        self.assertNotIn(b'.env', paths)
        self.assertNotIn(b'build/output', paths)
        self.assertFalse(any(b'/.git/' in p for p in paths))


if __name__ == '__main__':
    unittest.main()
