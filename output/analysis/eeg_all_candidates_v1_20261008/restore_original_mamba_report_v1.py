"""Restore only independently hash-proven original bytes; archive the real recheck."""
from pathlib import Path
import shutil
import sys
import time

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import storage as S

SOURCE = Path(__file__).resolve()
OLD_SHA = '2a499ff86ff460c54b16cf8167f880e4d12e3d05a2902300833284dee26ff99c'
NEW_SHA = 'ea8a12c1315b51c39a18f02e7955ddf58cb151c7e51a6652a8ffc36d3d3cfd3a'


def run():
    target = S.REPORT / 'eegmamba_official_gpu_check.json'
    recovered = S.REPORT / 'mamba_qualification_recovery/eegmamba_original_exact_1791548242887607600.json'
    provenance = S.REPORT / 'mamba_qualification_recovery/candidate_provenance_1791548242888580200.json'
    archive = S.OUT / 'assets/eegmamba_recheck_report_history/q4_recheck_1791547936287947400.json'
    if not (S.digest(target) == S.digest(archive) == NEW_SHA and S.digest(recovered) == OLD_SHA
            and S.digest(provenance) == 'ed095be7d2279478da8372521183487070c8da1bfeed6200f6911f2897ae964d'):
        raise ValueError('Expected real current/archived and exact original source-bound bytes differ')
    old, new = S.read(recovered), S.read(archive)
    # The known original hash, established before the recheck, authenticates
    # the whole restored document; no historical numeric value is inferred.
    shutil.copyfile(recovered, target)
    if S.digest(target) != OLD_SHA:
        raise ValueError('Original byte restoration failed')
    report = {'schema': 'mamba_recheck_report_exact_restore_v1', 'passed': True, 'created_at': S.now(),
              'source': S.desc(SOURCE), 'original_restored_report': S.desc(target),
              'original_exact_recovered_copy': S.desc(recovered), 'recovery_provenance': S.desc(provenance),
              'real_new_recheck_report_archived': S.desc(archive),
              'old_measured_seconds': old['seconds'], 'new_measured_seconds': new['seconds'],
              'new_peak_allocated_bytes': new['peak_gpu_allocated_bytes'],
              'new_recheck_is_engineering_not_formal_training': True,
              'new_engineering_optimizer_steps': new['engineering_optimizer_steps'],
              'new_engineering_encode_calls': new['gpu_encode_calls'],
              'old_source_bound_reports_and_new_real_cost_both_preserved': True,
              'original_report_unchanged_after_exact_byte_recovery': True,
              'new_recheck_primary_ledger': S.desc(S.OUT / 'engineering_attempts/EEGMamba/1791547890770467006_4c86e13ed7424198b3e9c439a6a85b91.json'),
              'nested_time_and_counter_reports_are_not_added_again': True,
              'no_model_checkpoint_or_raw_array_operations': True}
    path = S.OUT / 'progress_events' / ('mamba_recheck_report_restore_' + str(time.time_ns()) + '.json')
    S.save(path, report, immutable=True)
    print(S.desc(path))


if __name__ == '__main__':
    run()
