"""Run only registered label-free numerical analyses; reports are separate."""
from __future__ import annotations
import unittest
import traceback
from .common import *
from .audit import audit_one
from .scales import run_scale_audit

CORE=['__init__.py','common.py','quality.py','scales.py','audit.py','test_contracts.py','run.py']


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    before=history_check()
    records,folds=inputs()
    require(len(records)==37,'Cohort changed')
    locked_sources={p.relative_to(WORK).as_posix():digest(p) for p in [*(CODE/n for n in CORE),PLAN]}
    registration={'registered_utc':now(),'plan':desc(PLAN),'rules':RULES,'sources':locked_sources,
        'subjects':37,'diagnostic_labels_used':False,'augmentation':False,'classification_training':False,
        'shared_scale_fit':'outer-fold train identities only','old_artifacts_verified_before':before}
    if (OUT/'registration.json').exists():
        old=read_json(OUT/'registration.json')
        require(old['sources']==locked_sources and old['rules']==RULES,'Registered QC implementation changed')
    else:
        save(OUT/'registration.json',registration)
        save(OUT/'input_manifest_label_free.json',records)
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tridep_ccml.eeg_p2.test_contracts'))
    require(tests.wasSuccessful(),'QC regression failed')
    results=[]
    for record in records:
        results.append(audit_one(record))
        save(OUT/'status.json',{'state':'quality_audit','subjects_complete':len(results),'subjects_total':37,'updated_utc':now()})
    save(OUT/'subject_summary.json',[{'subject_id':r['subject_id'],
        'rms':r['filtered_19']['subject_rms'],
        'channel_scale_review':[c['channel'] for c in r['filtered_19']['channels'] if c['relative_scale_review']],
        'native_flat_channels':[c['channel'] for c in r['full_record_structure']['channels'] if c['flat_review']],
        'native_extreme_plateau_channels':[c['channel'] for c in r['full_record_structure']['channels'] if c['extreme_plateau_review']],
        'filtered_line_noise_channels':[c['channel'] for c in r['filtered_19']['channels'] if c['line_noise_review']],
        'filtered_high_frequency_channels':[c['channel'] for c in r['filtered_19']['channels'] if c['high_frequency_review']],
        'high_rms_windows_any_channel':sorted(set(i for c in r['filtered_19']['channels'] for i in c['high_rms_window_indices_0based'])),
        'ICA_metadata_present':r['processing_metadata']['ica_weights_nonempty'],
        'physical_calibration_fields':r['unit_reader_check']['calibration_fields']} for r in results])
    run_scale_audit(records,folds)
    from tridep_ccml.v2.extract_eeg import patches
    from scipy.io import loadmat
    import mne
    revision=read_json(REVISION)['selection']
    mat=loadmat(WORK/records[0]['mat']['path'])['eeg_mat']
    indices=[r['expected_zero_based_index'] for r in revision]
    names=[r['target_name'] for r in revision]
    native=patches(mat,indices,names)
    converted=patches(mat.astype(float)*1e-6,indices,names)*1e6
    np.testing.assert_allclose(native,converted,rtol=1e-7,atol=1e-7)
    raw=mne.io.RawArray(np.full((1,10),10e-6),mne.create_info(['demo'],250,'eeg'),verbose='ERROR')
    exported=raw.to_data_frame()['demo'].to_numpy()
    np.testing.assert_allclose(exported,10.,rtol=1e-12,atol=1e-12)
    save(OUT/'unit_numeric_checks.json',{'mne_version':mne.__version__,
        'read_eeglab_native_times_1e_minus6_verified_subjects':37,'fixed_blocks_per_subject':3,
        'to_data_frame_default_10_microvolt_signal_value':float(exported[0]),
        'linear_preprocessing_unit_roundtrip_subject':records[0]['subject_id'],
        'linear_preprocessing_unit_roundtrip_max_abs_error':float(np.max(np.abs(native-converted))),
        'implication':'Native->SI->linear filtering->native preserves numbers; model normalization is a separate contract.',
        'physical_gain_calibration_confirmed':False,
        'source_files':[desc(WORK/'S00/.venv/Lib/site-packages/mne/io/eeglab/eeglab.py'),desc(WORK/'S00/.venv/Lib/site-packages/mne/io/base.py')]})
    for name,sha in locked_sources.items():
        require(digest(WORK/name)==sha,'QC code changed during analysis')
    for record in records:
        for key in ['set','fdt','mat','cache']:
            require(digest(WORK/record[key]['path'])==record[key]['sha256'],'Input changed during QC')
    after=history_check()
    require(before==after,'Historical coverage changed')
    save(OUT/'verification.json',{'passed':True,'tests_passed':tests.testsRun,'subjects_checked':37,
        'FDT_MAT_exact':37,'R1_preprocessing_bitwise_replays':37,'train_fold_standardizer_replays':5,
        'old_artifacts_verified_before':before,'old_artifacts_verified_after':after,
        'original_inputs_unchanged':True,'labels_used_for_analysis':False,'classification_models_trained':0,
        'subjects_removed':0,'channels_removed':0,'augmentation':False,'registered_sources_unchanged':True})
    save(OUT/'status.json',{'state':'numeric_audit_complete_pending_report','subjects_complete':37,'updated_utc':now()})
    print('NUMERIC AUDIT COMPLETE; original signals, R1 and P1 unchanged.',flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException:
        save(OUT/'failure.json',{'at_utc':now(),'traceback':traceback.format_exc()})
        raise
