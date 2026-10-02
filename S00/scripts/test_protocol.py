import copy
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
from prepare import audit_folds
from metrics import metrics, select_weights

class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.folds=json.loads((Path(__file__).resolve().parents[1]/'manifests/fold_assignments.json').read_text())
    def test_official_folds(self):
        self.assertTrue(audit_folds(self.folds)['passed'])
    def test_original_38_rejected_for_active_experiment(self):
        original=json.loads((Path(__file__).resolve().parents[1]/'manifests/fold_assignments.official38.json').read_text())
        self.assertFalse(audit_folds(original)['passed'])
        self.assertTrue(audit_folds(original,expected_count=38)['passed'])
    def test_wrong_subject_even_with_37_people_rejected(self):
        ids={s for f in self.folds.values() for s in f['test']}
        old=sorted(ids)[0]
        replaced={n:{p:['02010004' if s==old else s for s in v] for p,v in f.items()} for n,f in self.folds.items()}
        self.assertFalse(audit_folds(replaced,expected_ids=ids)['passed'])
    def test_leak_rejected(self):
        self.folds['fold_1']['train'].append(self.folds['fold_1']['test'][0])
        self.assertFalse(audit_folds(self.folds)['passed'])
    def test_duplicate_rejected(self):
        self.folds['fold_2']['val'].append(self.folds['fold_2']['val'][0])
        self.assertFalse(audit_folds(self.folds)['passed'])
    def test_known_metrics(self):
        result=metrics([0,0,1,1],[0.1,0.8,0.2,0.9])
        self.assertEqual(result['macro_f1'],0.5)
        self.assertEqual(result['balanced_accuracy'],0.5)
        self.assertEqual(metrics([0,1],[0.5,0.8])['macro_f1'],1)
    def test_invalid_probability(self):
        with self.assertRaises(ValueError): metrics([1],[float('nan')])
    def test_test_selection_rejected(self):
        with self.assertRaises(ValueError): select_weights([{'split':'test'}])
    def test_fusion(self):
        rows=[{'split':'val','subject_id':str(y),'seed':11,'outer_fold':1,'label':y,'p_eeg':1-y,'p_speech':1-y,'p_text':y} for y in (0,1)]
        self.assertEqual(select_weights(rows)['weights'],[0,0,1])
        rows.append(rows[0])
        with self.assertRaises(ValueError): select_weights(rows)

class FeatureProvenanceTests(unittest.TestCase):
    def test_fitted_features_cannot_use_test_subjects(self):
        import numpy as np
        import run_b0
        from prepare import digest,save
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); save(root/'reports/modma_audit.json',{'passed':True})
            save(root/'protocol_resolution.json',{k:'fixture: synthetic data only' for k in ('speech_encoder','hubert_identity','asr_segments','audio_segmentation','eeg_units_channels','eeg_length_policy')})
            np.save(root/'x.npy',np.ones((1,1)))
            save(root/'features_manifest.json',[{'cohort_id':'fixture','folds_sha256':'fixture','seed':11,'outer_fold':1,'subject_id':'a','modality':'eeg','path':'x.npy','sha256':digest(root/'x.npy'),'producer_id':'fixture','fit_subject_ids':['c']}])
            folds={'fold_1':{'train':['a'],'val':['b'],'test':['c']}}
            cfg={'cohort_id':'fixture','folds_sha256':'fixture','branches':{'eeg':{'shape':[1,1]}}}
            lock={'cohort_id':'fixture','folds_sha256':'fixture','subject_ids':['a','b','c']}
            with patch.object(run_b0,'ROOT',root), patch.object(run_b0,'validate_active_cohort',return_value=lock):
                with self.assertRaisesRegex(ValueError,'leakage'):
                    run_b0.check_inputs(root,11,folds,cfg)
                rows=json.loads((root/'features_manifest.json').read_text())
                rows[0]['subject_id']='02010004'
                save(root/'features_manifest.json',rows)
                with self.assertRaisesRegex(ValueError,'cohort mismatch'):
                    run_b0.check_inputs(root,11,folds,cfg)

class SharedCohortTests(unittest.TestCase):
    def setUp(self):
        from cohort import ROOT,read,validate_active_cohort
        self.root=ROOT; self.lock=validate_active_cohort(require_data=True)
        self.folds=read(ROOT/self.lock['fold_file'])
        labels={r['subject_id']:r['label'] for r in read(ROOT/'manifests/subjects.json')}
        self.rows=[{'experiment':'S01-37','cohort_id':self.lock['cohort_id'],
                    'folds_sha256':self.lock['folds_sha256'],'seed':11,'outer_fold':int(n[-1]),
                    'split':'test','subject_id':sid,'label':labels[sid]}
                   for n,f in self.folds.items() for sid in f['test']]
    def test_followup_uses_same_37(self):
        from cohort import validate_prediction_records
        self.assertTrue(validate_prediction_records(self.rows,'S01-37')['passed'])
    def test_excluded_subject_prediction_rejected(self):
        from cohort import validate_prediction_records
        self.rows[0]['subject_id']='02010004'
        with self.assertRaisesRegex(ValueError,'locked cohort'):
            validate_prediction_records(self.rows,'S01-37')
    def test_mismatched_fold_cache_rejected(self):
        from cohort import validate_prediction_records
        self.rows[0]['folds_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'provenance'):
            validate_prediction_records(self.rows,'S01-37')
    def test_missing_test_subject_rejected(self):
        from cohort import validate_prediction_records
        with self.assertRaisesRegex(ValueError,'exactly 37'):
            validate_prediction_records(self.rows[:-1],'S01-37')
    def test_duplicate_test_subject_rejected(self):
        from cohort import validate_prediction_records
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            validate_prediction_records(self.rows+[self.rows[0]],'S01-37')
    def test_stale_38_person_audit_rejected(self):
        from cohort import validate_active_cohort
        from prepare import save
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ['configs/cohort.json','manifests/fold_assignments.json','manifests/fold_assignments.official38.json','manifests/subjects.json']:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(self.root/name,p)
            save(root/'reports/modma_audit.json',{'passed':True,'cohort':38,'wav_count':1102})
            with self.assertRaisesRegex(ValueError,'stale'):
                validate_active_cohort(root,require_data=True)

if __name__=='__main__': unittest.main(verbosity=2)
