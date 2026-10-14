"""Stdlib-only V2 provenance/gate regressions; no actual matrix or GPU run."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name('qualify_training_sync_v2.py')
spec = importlib.util.spec_from_file_location('_training_sync_v2_test', SOURCE)
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


class WrapperTests(unittest.TestCase):
    def test_old_source_exact_and_math_functions_still_original_code(self):
        self.assertEqual(hashlib.sha256(M.DELEGATE.read_bytes()).hexdigest(), M.DELEGATE_SHA)
        for name in ('run_matrix','summarize','matrix_plan','fingerprint','forbid_optimizer_steps','qualify','main'):
            function=getattr(M.D,name)
            self.assertEqual(getattr(function,'__wrapped__',function).__code__.co_filename,str(M.DELEGATE))
        self.assertEqual(M.D.SOURCE,M.SOURCE)
        self.assertEqual(M.D.GATE_SOURCE,M.GATE_SOURCE)

    def test_exact_fixed96_budget_and_alternating_orders(self):
        plan=M.D.matrix_plan()
        self.assertEqual(len(plan),12)
        self.assertEqual(M.D.FIXED_COUNTS,{'forward':96,'encoder_forward':96,'encoder_windows':768,
                                          'backward':96,'optimizer_step':0,'optimizer_step_stub':12})
        self.assertEqual(sum(row['warmup'] for row in plan),4)
        self.assertEqual([r['variant'] for r in plan[:6]],['original','candidate','candidate','original','original','candidate'])

    def identity(self):
        split={'train':['train'+str(i) for i in range(8)],'val':['val'],'test':['test']}
        fit={'condition':M.D.CONDITION,'seed':22,'fold':5,'stage':'3','fit_id':M.D.CONDITION+'/seed_22/fold_5',
             **{k+'_ids':v for k,v in split.items()}}
        proof={'schema':'sync_qualification_safe_boundary_v2','passed':True,'condition':M.D.CONDITION,'seed':22,'fold':5,
               'checkpoint':{'path':(M.D.OUT/'runs'/fit['fit_id']/'latest.pt').relative_to(M.WORK).as_posix()}}
        return proof,{'folds':{'fold_5':split}},fit,[{'epoch':1}]

    def test_actual_v2_schema_required_and_roles_retained(self):
        args=self.identity()
        self.assertEqual(M.D.validate_fixed_identity(*args)[0],args[2]['train_ids'])
        for change in ({'schema':'sync_qualification_safe_boundary_v1'},{'seed':11},{'passed':False}):
            with self.assertRaises(ValueError):M.D.validate_fixed_identity({**args[0],**change},*args[1:])
        bad={**args[2],'test_ids':['train0']}
        with self.assertRaises(ValueError):M.D.validate_fixed_identity(args[0],args[1],bad,args[3])

    def test_provenance_is_new_with_delegated_source_and_input_unchanged(self):
        for schema in ('training_sync_qualification_attempt_v1','training_sync_qualification_v1','training_sync_qualification_pending_v1'):
            original={'schema':schema,'seconds':None,'optimizer_updates':0}
            value=M.provenance(original)
            self.assertEqual(value['schema'],schema[:-2]+'v2')
            self.assertEqual(value['source'],M.descriptor(M.SOURCE))
            self.assertEqual(value['delegated_training_qualification_source'],M.descriptor(M.DELEGATE))
            self.assertEqual(value['actual_gate_source'],M.descriptor(M.GATE_SOURCE))
            self.assertEqual(value['qualification_execution_version'],2)
            self.assertEqual(original,{'schema':schema,'seconds':None,'optimizer_updates':0})
        self.assertEqual(M.provenance({'id':'lease'}),{'id':'lease'})

    def test_pending_writes_truthful_v2_without_gate_import_or_tensor_import(self):
        with tempfile.TemporaryDirectory(dir=M.SOURCE.parent,prefix='training_v2_test_') as temp:
            with patch.object(M.D,'REPORT',Path(temp)),patch.object(M.D,'gate_module',side_effect=AssertionError('gate should not run')):
                self.assertEqual(M.main([]),0)
            saved=json.loads(next(Path(temp).glob('*.json')).read_text(encoding='utf-8'))
            self.assertEqual(saved['schema'],'training_sync_qualification_pending_v2')
            self.assertEqual(saved['GPU'],0);self.assertEqual(saved['torch_imports'],0)
            self.assertFalse(saved['passed'])

    def test_preflight_failure_records_actual_v2_zero_work_and_no_lease(self):
        class RejectGate:
            OWNER=None
            def preflight(self,*args):raise ValueError('synthetic boundary rejection')
        with tempfile.TemporaryDirectory(dir=M.SOURCE.parent,prefix='training_v2_test_') as temp:
            root=Path(temp);gate=RejectGate();gate.OWNER=root/'owner.json'
            with patch.object(M.D,'OUT',root),patch.object(M.D,'OWNER',gate.OWNER),patch.object(M.D,'gate_module',return_value=gate):
                with self.assertRaisesRegex(ValueError,'synthetic boundary rejection'):M.qualify({'path':'fake.json','sha256':'0'*64})
            saved=json.loads(next(root.rglob('training_sync_attempt_*.json')).read_text(encoding='utf-8'))
            self.assertEqual(saved['schema'],'training_sync_qualification_attempt_v2');self.assertEqual(saved['status'],'failed')
            self.assertFalse(saved['gpu_stage_entered']);self.assertEqual(saved['backward_started'],0)
            self.assertEqual(saved['optimizer_step_started'],0);self.assertEqual(saved['model_constructions_started'],0)
            self.assertFalse(gate.OWNER.exists())


if __name__=='__main__':unittest.main(verbosity=2)
