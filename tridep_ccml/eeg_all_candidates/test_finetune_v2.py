"""Synthetic CPU regression for LUNA cached fixed frequencies and v2 evidence."""
from __future__ import annotations

from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import finetune_v2 as F
from .adapters import luna as L1, luna_v2 as L2


class FinetuneVersion2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_author_fixed_frequency_stays_frozen_after_warmup_cache_and_repeated_finetune(self):
        model,audit=F.build_model('LUNA-lastblock-finetune',device='cpu')
        self.assertEqual(audit['native_dimension'],256)
        self.assertEqual(len(model.selected_names),17)
        self.assertFalse(any(n.endswith('rotary_emb.freqs') for n in model.selected_names))
        rotary=model.encoder.blocks[-1].attn.rotary_emb
        original=rotary.freqs.detach().clone()
        raw=np.random.default_rng(20261008).normal(0,20,(1,8,19,1000)).astype(np.float32)
        F.V1.set_train_phase(model,1)
        loss=torch.nn.functional.cross_entropy(model(raw),torch.tensor([1]));loss.backward()
        self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))
        self.assertEqual(rotary.cached_freqs_seq_len,32)
        for epoch in (11,12,13):
            model.zero_grad(set_to_none=True);F.V1.set_train_phase(model,epoch)
            torch.nn.functional.cross_entropy(model(raw),torch.tensor([1])).backward()
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad))
            self.assertFalse(rotary.learned_freq);self.assertFalse(rotary.freqs.requires_grad)
            self.assertIsNone(rotary.freqs.grad);self.assertTrue(torch.equal(original,rotary.freqs))

    def test_frozen_original_forward_and_readout_are_identical(self):
        v1,_=L1.load_encoder('cpu',trainable=False);v2,_=L2.load_encoder('cpu',trainable=False)
        raw=np.random.default_rng(4).normal(size=(2,19,1000)).astype(np.float32)
        self.assertTrue(np.array_equal(L1.adapt_raw(raw),L2.adapt_raw(raw)))
        values=torch.from_numpy(L1.adapt_raw(raw))
        with torch.inference_mode():
            self.assertTrue(torch.equal(L1.encode_windows(v1,values),L2.encode_windows(v2,values)))

    def test_selected_warmup_encoder_is_not_claimed_as_terminal_update(self):
        initial={'encoder.weight':torch.tensor([1.]),'head.weight':torch.tensor([1.])}
        selected={'encoder.weight':torch.tensor([1.]),'head.weight':torch.tensor([2.])}
        terminal={'encoder.weight':torch.tensor([3.]),'head.weight':torch.tensor([4.])}
        meta={'best_epoch':7,'parameter_changed_after_training':{'encoder.weight':True,'head.weight':True}}
        value=F.selected_checkpoint_disclosure(meta,initial,selected,terminal)
        self.assertEqual(value['selected_checkpoint_phase'],'head_only_warmup')
        self.assertEqual(value['selected_encoder_changed_parameter_count'],0)
        self.assertEqual(value['selected_head_changed_parameter_count'],1)
        self.assertEqual(value['terminal_encoder_changed_parameter_count'],1)
        self.assertEqual(value['selected_encoder_initial_state_sha256'],value['selected_encoder_state_sha256'])
        self.assertNotEqual(value['terminal_encoder_state_sha256'],value['selected_encoder_state_sha256'])

    def test_attempt_failure_cost_and_independent_probe_paths_are_durable(self):
        F.S.REPORT.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='ft_v2_cpu_',dir=F.S.REPORT) as temporary:
            root=Path(temporary).resolve();self.assertTrue(root.is_relative_to(F.S.REPORT.resolve()))
            with patch.object(F.S,'OUT',root):
                first=F.Attempt('synthetic','engineering');second=F.Attempt('synthetic','engineering')
                self.assertNotEqual(first.folder/'resume_probe.pt',second.folder/'resume_probe.pt')
                def failure():raise ValueError('synthetic forward failure')
                with self.assertRaisesRegex(ValueError,'synthetic'):
                    first.observe('forward',failure)()
                first.finish(ValueError('synthetic forward failure'))
                written=F.S.read(first.path)
                self.assertEqual(written['forward_started'],1);self.assertEqual(written['forward_returned'],0)
                self.assertEqual(written['backward_started'],written['optimizer_step_started'],0)
                self.assertEqual(written['status'],'failed');self.assertEqual(written['failure_type'],'ValueError')

    def test_early_prepare_failure_is_logged_before_data_or_checkpoint_access(self):
        F.S.REPORT.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='ft_v2_failure_',dir=F.S.REPORT) as temporary:
            root=Path(temporary).resolve();self.assertTrue(root.is_relative_to(F.S.REPORT.resolve()))
            with patch.object(F.S,'OUT',root),patch.object(torch.cuda,'is_available',return_value=False), \
                 patch.object(torch.cuda,'empty_cache'),patch.object(F.V1,'_raw_sources') as data:
                with self.assertRaisesRegex(ValueError,'CUDA'):
                    F.prepare_condition('LUNA-lastblock-finetune')
                data.assert_not_called()
            attempts=list(root.glob('finetune_v2_attempts/*/e_*/attempt.json'))
            self.assertEqual(len(attempts),1)
            value=F.S.read(attempts[0])
            self.assertEqual(value['status'],'failed')
            self.assertEqual(value['forward_started'],value['backward_started'],0)
            self.assertEqual(value['optimizer_step_started'],0)
            self.assertGreaterEqual(value['seconds'],0.)

    def test_synthetic_prepare_records_exact_calls_and_fresh_probe_on_retry(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__();self.frozen=torch.nn.Linear(1,4);self.last=torch.nn.Linear(4,4)
            def forward(self,value):return self.last(self.frozen(value.mean(dim=(1,2)).unsqueeze(-1)))
        adapter=types.SimpleNamespace(adapt_raw=lambda value:value,encode_windows=lambda model,value:model(value))
        descriptor=F.S.desc(Path(__file__))
        raw={'synthetic_train':np.random.default_rng(1).normal(size=(60,19,1000)).astype(np.float32)}
        split={'train':['synthetic_train'],'val':['synthetic_val'],'test':['synthetic_test']}
        def build(*args,**kwargs):
            encoder=Encoder();names=[n for n,_ in encoder.named_parameters() if n.startswith('last.')]
            model=F.V1.RawSubjectModel(encoder,F.V1.SubjectBagHead(4),adapter,names,5,8)
            model.forward=F._TRACE.observe('forward',model.forward);F._TRACE.bind_adapter(adapter)
            return model,{'checkpoint':descriptor}
        F.S.REPORT.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='ft_v2_prepare_',dir=F.S.REPORT) as temporary:
            root=Path(temporary).resolve();self.assertTrue(root.is_relative_to(F.S.REPORT.resolve()))
            with patch.object(F.S,'OUT',root),patch.object(torch.cuda,'is_available',return_value=True), \
                 patch.object(torch.cuda,'empty_cache'),patch.object(torch.cuda,'memory_allocated',return_value=0), \
                 patch.object(torch.cuda,'max_memory_allocated',return_value=0),patch.object(torch.cuda,'reset_peak_memory_stats'), \
                 patch.object(F.V1.R,'load_registration'),patch.object(F.V1.R,'lock_condition'),patch.object(F.V1.R,'record_fit_status'), \
                 patch.object(F.V1.C,'context',return_value=({}, {'fold_1':split},{'synthetic_train':1})),patch.object(F.V1.C,'runtime'), \
                 patch.object(F.V1,'_raw_sources',return_value=(raw,{'synthetic_train':descriptor})),patch.object(F,'build_model',side_effect=build), \
                 patch.object(F,'implementation_sources',return_value=[descriptor]):
                first=F.prepare_condition('LUNA-lastblock-finetune');second=F.prepare_condition('LUNA-lastblock-finetune')
            self.assertNotEqual(first['resume_probe']['path'],second['resume_probe']['path'])
            for contract in (first,second):
                ledger=F.S.read(F.S.WORK/contract['engineering_attempt_path'])
                self.assertEqual(ledger['forward_started'],2);self.assertEqual(ledger['forward_returned'],2)
                self.assertEqual(ledger['encode_started'],2);self.assertEqual(ledger['encode_returned'],2)
                self.assertEqual(ledger['backward_started'],2);self.assertEqual(ledger['backward_returned'],2)
                self.assertEqual(ledger['optimizer_step_started'],0);self.assertEqual(ledger['optimizer_step_returned'],0)
                self.assertEqual(ledger['encoder_windows_returned'],16)
                self.assertEqual(ledger['status'],'complete')

    def test_hooks_restore_and_canonical_schedules_are_delegated(self):
        before=(F.V1.build_model,F.V1._implementation_sources,F.S.save,torch.Tensor.backward,torch.optim.AdamW.step)
        with F.activate():
            self.assertIs(F.V1.build_model,F.build_model)
            self.assertIs(F.V1._implementation_sources,F.implementation_sources)
        self.assertEqual(before,(F.V1.build_model,F.V1._implementation_sources,F.S.save,torch.Tensor.backward,torch.optim.AdamW.step))
        self.assertEqual(F.V1.P.POLICIES['finetune'].warmup_epochs,10)
        self.assertEqual(F.V1.P.POLICIES['finetune'].backbone_warmup_epochs,5)
        self.assertEqual(F.V1.P.learning_rate_multiplier('finetune',10,parameter_group='backbone'),0.)
        self.assertEqual(F.V1.P.learning_rate_multiplier('finetune',11,parameter_group='backbone'),.2)


if __name__=='__main__':unittest.main()
