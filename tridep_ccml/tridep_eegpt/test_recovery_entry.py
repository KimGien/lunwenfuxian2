"""Pure tests of provenance gates and the sole evaluation dependency bridge."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import recovery as R
from . import run


class RecoveryEntryTests(unittest.TestCase):
    def test_training_rejects_non_readonly_gate_before_any_write(self):
        execution = {'path': 'recovery/registration.json', 'sha256': 'a'}
        receipt = {'passed': True, 'execution_registration': execution,
                   'source': {'path': 'verify', 'sha256': 'b'},
                   'mode': 'registration_only', 'read_only': False}
        with patch.object(R, 'check_registration', return_value={}), \
             patch.object(R, 'execution_registration', return_value=execution), \
             patch.object(R, 'PLAIN_READ', return_value=receipt), \
             patch.object(R.C, 'desc', return_value=receipt['source']), \
             patch.object(R.C, 'save') as saved:
            with self.assertRaises(ValueError):
                R.train()
            saved.assert_not_called()

    def test_direct_fit_guard_rejects_false_source(self):
        receipt = {'passed': True, 'mode': 'registration_only', 'read_only': True,
                   'execution_registration': {'sha256': 'a'}, 'source': {'sha256': 'wrong'}}
        with patch.object(R, 'check_registration', return_value={}), \
             patch.object(R, 'execution_registration', return_value={'sha256': 'a'}), \
             patch.object(R, 'PLAIN_READ', return_value=receipt), \
             patch.object(R.C, 'desc', return_value={'sha256': 'actual'}):
            with self.assertRaises(ValueError):
                R.guard_training()

    def test_exact_path_bridge_both_import_aliases_and_exception_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            failed = out / 'actions/fit_all.json'
            completed = out / 'actions/fit_all_recovery.json'
            failed.parent.mkdir()
            failed.write_text(json.dumps({'status': 'failed'}), encoding='utf-8')
            execution = {'path': 'new', 'sha256': 'a'}
            completed.write_text(json.dumps({'status': 'complete', 'execution_registration': execution}), encoding='utf-8')
            before = failed.read_bytes()
            old_failure = {'path': str(failed), 'sha256': 'b'}
            def raw(path):
                return json.loads(Path(path).read_text(encoding='utf-8'))
            def live(path):
                if Path(path).resolve() in (failed.resolve(), completed.resolve()):
                    return raw(path)
                return {'ordinary': str(path)}
            with patch.object(R.C, 'OUT', out), \
                 patch.object(R, 'check_registration', return_value={'failed_action': old_failure}), \
                 patch.object(R, 'execution_registration', return_value=execution), \
                 patch.object(R, 'PLAIN_READ', side_effect=raw), \
                 patch.object(R.C, 'desc', return_value=old_failure), \
                 patch.object(R.C, 'read', live), patch.object(run, 'read', live):
                with self.assertRaisesRegex(RuntimeError, 'synthetic'):
                    with R.evaluation_bridge():
                        self.assertEqual(R.C.read(failed)['status'], 'complete')
                        self.assertEqual(run.read(failed)['status'], 'complete')
                        self.assertEqual(R.C.read(out / 'ordinary'), {'ordinary': str(out / 'ordinary')})
                        self.assertEqual(failed.read_bytes(), before)
                        raise RuntimeError('synthetic')
                self.assertIs(R.C.read, live)
                self.assertIs(run.read, live)
                self.assertEqual(R.C.read(failed)['status'], 'failed')
                self.assertEqual(failed.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
