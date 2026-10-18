import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/rag_rrt_real_validation'))
from rag_rrt_real_validation.entry_points import ENTRY_POINTS, verify_entry_points


class EntryPointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.prefixes = {name: Path(self.temp.name)/name for name in ENTRY_POINTS}
        for package, (name, _) in ENTRY_POINTS.items():
            parent = self.prefixes[package]/'lib'/package
            parent.mkdir(parents=True)
            (parent/name).touch()

    def test_empty_files_block_startup_and_can_be_repaired(self):
        with self.assertRaisesRegex(RuntimeError, 'empty Python entry point'):
            verify_entry_points(self.prefixes)
        results = verify_entry_points(self.prefixes, repair=True)
        self.assertTrue(all(item['bytes'] > 50 for item in results))
        for item in results:
            self.assertTrue(Path(item['path']).read_bytes().startswith(b'#!'))
            self.assertTrue(os.access(item['path'], os.X_OK))
        self.assertEqual(results, verify_entry_points(self.prefixes))

    def test_incremental_verification_preserves_valid_wrappers(self):
        first = verify_entry_points(self.prefixes, repair=True)
        before = {item['path']: Path(item['path']).read_bytes() for item in first}
        verify_entry_points(self.prefixes, repair=True)
        self.assertEqual(before, {p: Path(p).read_bytes() for p in before})

    def test_no_fake_install_created_for_missing_package(self):
        self.prefixes['rag_rrt_backend_py'] = Path(self.temp.name)/'missing'
        with self.assertRaisesRegex(RuntimeError, 'not installed'):
            verify_entry_points(self.prefixes, repair=True)


if __name__ == '__main__':
    unittest.main()
