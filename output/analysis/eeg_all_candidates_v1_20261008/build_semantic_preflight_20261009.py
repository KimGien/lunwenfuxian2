"""Source/text-only semantic preparation; deliberately cannot certify completion."""
from pathlib import Path
import hashlib
import json
import time
from datetime import datetime, timezone, timedelta

WORK = Path(__file__).resolve().parents[3]
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
REPORT = WORK / 'output/analysis/eeg_all_candidates_v1_20261008'
MOD = 'tridep_ccml/eeg_all_candidates/'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def desc(path):
    path = Path(path).resolve()
    assert path.is_relative_to(WORK)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': digest}


def check(record):
    assert desc(WORK / record['path']) == record, record['path']
    return record


def location(name, needle, lines=1):
    path = WORK / (MOD + name if not name.startswith(('output/', 'tridep_ccml/')) else name)
    content = path.read_text(encoding='utf-8').splitlines()
    matches = [i + 1 for i, line in enumerate(content) if needle in line]
    assert len(matches) == 1, (name, needle, matches)
    start = matches[0]
    return {'source': desc(path), 'line': start, 'end_line': min(start + lines - 1, len(content)),
            'anchor': needle}


def claim(text, observed, *locations):
    return {'claim': text, 'observed': observed, 'evidence_locations': list(locations)}


def item(ident, finding, checks, extra, pending):
    records = list(extra)
    for row in checks:
        records.extend(loc['source'] for loc in row['evidence_locations'])
    evidence = sorted({r['path']: r for r in records}.values(), key=lambda r: r['path'])
    for record in evidence:
        check(record)
    return {'id': ident, 'conclusion': 'preflight_only_final_verification_pending',
            'finding': finding, 'checks': checks, 'evidence': evidence, 'final_checks_pending': pending}


def main():
    started = datetime.now(timezone(timedelta(hours=8))).isoformat()
    registration = read(OUT / 'registration.json')
    q3 = read(OUT / 'continuation_queue_v3/registration.json')
    assert len(q3['sources']) == 47
    for record in q3['sources']:
        check(record)
    engineering = []
    supplemental = []
    reg_desc = desc(OUT / 'registration.json')
    first = next(r for r in registration['fits'] if r['seed'] == 11 and r['fold'] == 1)
    train_people = first['train_ids']
    for path in sorted((OUT / 'engineering').glob('*.json')):
        value = read(path)
        contract = value['contract']
        canonical = hashlib.sha256(json.dumps(contract, ensure_ascii=False, sort_keys=True,
                                  separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        assert value['registration'] == reg_desc and value['configuration_sha256'] == canonical
        assert all(contract.get(key) is True for key in ('forward_checked', 'gradient_checked', 'resume_checked', 'training_only_engineering_checked'))
        trainable, frozen = contract['trainable_parameter_names'], contract['frozen_parameter_names']
        assert len(trainable) == len(set(trainable)) and not set(trainable) & set(frozen)
        textual_bindings = [r for r in contract['implementation_sources'] + contract['weight_sources']
                            if Path(r['path']).suffix.lower() in ('.json', '.py')]
        for record in textual_bindings:
            check(record)
        engineering.append({'condition': value['condition'], 'artifact': desc(path), 'contract_hash_verified': True,
                            'true_engineering_flags': True, 'trainable_names_count': len(trainable),
                            'frozen_names_count': len(frozen), 'textual_bindings_verified': textual_bindings,
                            'binary_weights_or_checkpoints_not_loaded_by_this_review': True,
                            'probe_subject_ids': contract.get('probe_subject_ids'),
                            'existing_probe_resource_fields': {k: contract[k] for k in ('seconds', 'forward_backward_seconds', 'peak_allocated_bytes') if k in contract}})
        condition = value['condition']
        if condition.endswith('-frozen-smallhead') or condition.endswith('-originalCNN-300ep'):
            paired = condition.endswith('-originalCNN-300ep')
            n = 16 if paired else 8
            assert contract['probe_subject_ids'] == train_people[:n]
            supplemental.append({'condition': condition, 'existing_engineering_lock': desc(path),
                                 'historical_probe_seconds': contract.get('forward_backward_seconds', contract.get('seconds')),
                                 'historical_probe_peak_allocated_bytes': contract.get('peak_allocated_bytes'),
                                 'missing_historical_fields': ['peak_allocated_bytes'] + ([] if paired else ['seconds']),
                                 'future_independent_probe_subject_ids': train_people[:n],
                                 'future_independent_probe_input': contract['input_contract'],
                                 'future_probe_head_source': desc(WORK / MOD / ('paired_replay.py' if paired else 'frozen_heads.py')),
                                 'feature_manifest': contract['weight_sources'][0],
                                 'future_model_class': 'original EEGConvPoolClassifier' if paired else 'SubjectHead(native_dimension)',
                                 'future_optimizer_for_roundtrip_only': 'Adamax lr=.005 wd=0' if paired else 'AdamW lr=.001 wd=.001',
                                 'optimizer_steps_allowed': 0, 'gradient_passes': 1, 'seed': 11, 'fold': 1})
    engineering_desc = [r['artifact'] for r in engineering]
    manifests = [desc(path) for path in sorted((OUT / 'frozen_features').glob('*/manifest.json'))]
    download = read(OUT / 'assets/download_manifest.json')
    sources = {name: {'repository': row['repository'], 'revision': row['revision']}
               for name, row in download['sources'].items()}
    pause_path = OUT / 'pause_events/1791460680736346200/pause_complete.json'
    resume_path = OUT / 'resume_events/1791506856841705600.json'
    advanced_path = OUT / 'resume_events/resumed_1791506979532166700.json'
    pause, resume, advanced = read(pause_path), read(resume_path), read(advanced_path)
    check(resume['pause_record']); check(resume['owner_archived'])
    assert resume['owner_archived']['sha256'] == pause['owner']['sha256']
    for row in resume['preserved_artifacts']:
        check(row['preserved_copy'])
        assert row['preserved_copy']['sha256'] == row['original_at_resume']['sha256']
    assert resume['old_exact_pids_and_related_experiment_python_processes_observed'] == []
    assert resume['acknowledged_updates_not_in_checkpoint'] == 108 - 104 == 4
    assert resume['acknowledged_backward_calls_not_in_checkpoint'] == 729 - 702 == 27
    assert advanced['training_has_advanced_beyond_paused_epoch'] is True
    lifecycle = []
    for version in ('v2', 'v3'):
        path = OUT / ('continuation_queue_' + version) / 'handoff_complete.json'
        proof = read(path)
        assert proof['passed'] and proof['launcher_exit_code'] == proof['gpu_child_exit_code'] == 0
        for record in proof['protected_sources']:
            check(record)
        lifecycle.append(desc(path))
    h3_state_path = OUT / 'continuation_queue_v3/handoff/state.json'
    h3_state = read(h3_state_path)
    archived_old_status = next(row['preserved_copy'] for row in resume['preserved_artifacts'] if row['preserved_copy']['path'].endswith('/old_attempt_status.json'))
    archived_old_log = next(row['preserved_copy'] for row in resume['preserved_artifacts'] if row['preserved_copy']['path'].endswith('/old_attempt_process.log'))
    old_attempt = OUT / 'continuation_queue_v3/attempts/1791458796212822000-eeg2rep-supervised-fit'
    assert desc(old_attempt / 'status.json')['sha256'] == archived_old_status['sha256']
    assert desc(old_attempt / 'process.log')['sha256'] == archived_old_log['sha256']
    rows = []
    rows.append(item('R04', 'Pinned adapter source implements explicit physical input adaptation; existing locks/manifests bind the executed frozen/scratch/SSL routes. Mamba and all fine-tune route evidence must be revisited after actual qualification and fitting.', [
        claim('Physical resampling and duration are explicit.', 'EEGPT uses polyphase32/25 to256Hz, crops1024 samples=4s and /1000; LUNA uses32/25 to1280 samples=5s, unbiased per-channel z score and40-sample patches. LaBraM crops exactly4/5s at200Hz and /100.', location('adapters/eegpt.py','def adapt_raw(raw):',5), location('adapters/luna.py','def adapt_raw(raw):',11), location('adapters/labram.py','def adapt_raw(raw, seconds=5):',7)),
        claim('Channel identities and nominal coordinates are retained.', 'CSBrain retains FCz/CPz central-region identity; REVE checks all19 named position-bank entries against standard1005 metre/MRI coordinates and marks them nominal; LUNA obtains official nominal channel locations and validates19x3.', location('adapters/csbrain.py','REGIONS =',15), location('adapters/reve.py','def nominal_positions()',32), location('adapters/luna.py','locations = module.ChannelEmbeddings',5)),
        claim('BIOT differences raw channels before independent normalization.', 'Canonical aliases locate each of16 registered bipolar endpoints; values are raw[a]-raw[b], then normalized by that resulting channel95th-percentile absolute value plus1e-8; perturb=False.', location('adapters/biot.py','def adapt_raw(raw):',5), location('adapters/biot.py','output = model(tensor, n_channel_offset=0, perturb=False)',2)),
        claim('CodeBrain and Mamba use the actual author architectures.', 'CodeBrain strictly loads SSSM, keeps EEGSSM feature path and excludes tokenizer heads; Mamba strictly loads12 real Mamba2 blocks, then replaces already-loaded reconstruction projection by Identity. Mamba source asserts Linux/CUDA but this is not executed qualification evidence.', location('adapters/codebrain.py','def load_encoder(device=',27), location('adapters/eegmamba.py','def load_encoder(device=',31)),
        claim('Scratch and fold-local SSL are correctly distinct weight routes.', 'Conformer permits exactly one22-to19 spatial-kernel adaptation and fits scalar normalization on declared train people. EEG2Rep has no external pretrained state; supervised binding requires same seed/fold train/val and genuinely updated selected SSL checkpoint.', location('adapters/eegconformer.py','def author_classes():',24), location('adapters/eegconformer.py','def fit_train_statistics(',9), location('supervised.py','def validate_ssl_payload(',25)),
        claim('Weight and source checks precede model loading.', 'Shared author support validates pinned Python files and full weight bytes/SHA; EEGPT verifies official encoder strict coverage and LaBraM reports explicit implementation and weight descriptors.', location('adapters/_author_support.py','def verified_resources(',15), location('tridep_ccml/tridep_eegpt/assets.py','def load_frozen_eegpt(',46), location('adapters/labram.py','def load_encoder(device=',17)),
    ], engineering_desc + manifests + [desc(OUT / 'assets/download_manifest.json')], ['Inspect actual Mamba CUDA check/export/runtime and both Mamba locks.', 'Inspect all9FT locks and135 completed-fit adapter/weight bindings; do not assume frozen-route success implies FT qualification.', 'Final full360 deep audit must prove every initializer/checkpoint/role; this preflight has loaded no tensors.']))
    rows.append(item('R06', 'Actual successful qualification is present for the listed existing engineering locks. Static code or CPU tests cannot qualify pending Mamba/FT routes. Historical head-probe timing/peak gaps are explicitly preserved and require an independent supplemental measurement if strict plan coverage is desired.', [
        claim('Formal routes require source-bound engineering locks.', 'Registry lock creation requires four true probe flags, disjoint exact trainable/frozen names, source hashes and weight provenance. Runtime _condition_lock checks identity/configuration/source bytes before ready/training. Flag existence alone does not independently establish actual execution.', location('registry.py','def lock_condition(',23), location('registry.py','def _condition_lock(',10)),
        claim('Existing frozen and paired checks perform real train-only forward/backward and resume roundtrip.', 'Frozen head checks use fold1 first8 train people and one backward; paired checks use first16, originalCNN and Adamax. Neither executes an optimizer step. Historical frozen head locks lack timing/peak; paired records timing but no peak.', location('frozen_heads.py','def prepare_condition(condition):',51), location('paired_replay.py','# Resource configuration uses training people only, with no optimizer step.',44)),
        claim('Completed SSL qualified the actual64-window loss.', 'Sealed SSL settings report logical_batch64, BF16 encoder/FP32 complete loss, 64 returned windows, one gradient pass, zero optimizer/EMA updates, peak2476714496 bytes and measured3.2726803seconds. Qualification is a distinct event, not15 completed SSL fits by itself.', location('ssl.py','def prepare(*, logical_batch=64',83), location('ssl_v2.py','def ssl_loss(output):',9)),
        claim('Future F2 qualification is real CUDA and records attempts before risky work.', 'Attempt exists before raw/model load and logs started/returned calls; preparation uses train-only phase1/11, two actual forward/backwards, zero optimizer steps, full-state serialization and resource fields before lock creation.', location('finetune_v2.py','class Attempt:',45), location('finetune_v2.py','def prepare_condition(condition,',81)),
        claim('Mamba check is separate from formal qualification/fits.', 'Linux version-gated synthetic check records4encode calls/1SGDstep as engineering0formal, then native feature export and train-only F2 qualification remain separate. Actual completed records are still required.', location('linux_entry.py','def _check_with_attempt_ledger(',46), location('check_eegmamba_gpu.py','def main():',56)),
    ], engineering_desc + [desc(OUT / 'ssl/engineering_settings.json')], ['Wait for real Mamba check and all9FT qualification attempt logs, names, scopes, resume probe and peak/duration.', 'Re-enumerate resource-field gaps after remaining engineering jobs; Mamba frozen head uses the same uninstrumented frozen prepare source.', 'Run separate0optimizer supplemental probes only after queue fully exits; retain old unknown values and separate costs.']))
    rows.append(item('R10', 'The inspected executed runners implement the fixed optimizers, scheduler, train/validation-only selection and full test gate. Final histories and complete-state optimizer evidence remain necessary for every future fit.', [
        claim('Optimizer/scheduler budgets are fixed in the sealed protocol.', 'AdamW policies specify frozen/FT/scratch/SSL/SSL-supervised LR/wd; old diagnostic remains Adamax. Warmup/cosine floor0.1 and extension hold are deterministic; best CE needs1e-4 improvement and earlier ties survive.', location('protocol.py','POLICIES =',14), location('protocol.py','def training_decision(',27), location('protocol.py','def learning_rate_multiplier(',20)),
        claim('FT head warmup and backbone warmup occur in code.', 'First10 epochs freeze the encoder, then selected modules enable; optimizer has explicit head/backbone groups and protocol multipliers. Logical8-person subject CE divides by actual people before clipping1 and one step. E2 distinguishes terminal updates from selected head-only best.', location('finetune.py','def set_train_phase(',20), location('finetune.py','def optimizer_for(',12), location('finetune.py','def subject_update(',25), location('evaluate_v2.py','def selection_evidence(',31)),
        claim('Validation cannot access outer-test people or perturb training state.', '_raw_sources only permits train+val; diagnostics loops exactly train,val, forbids test membership, snapshots/restores RNG and checks unchanged model state. P.training_decision receives validationCE only.', location('finetune.py','def _raw_sources(',14), location('finetune.py','def diagnostics(',29), location('protocol.py','def training_decision(',5)),
        claim('All360 gate precedes evaluation and source version changes do not tune on results.', 'Continuation evaluation tail invokes registry test gate; Q3 preserves original job identities/dependencies and E2 routing; no LR grid, threshold search or score-dependent candidate pruning is introduced in reviewed wrappers.', location('continuation_queue.py','def evaluate_tail(',28), location('continuation_queue_v3.py','def jobs():',12), location('ssl_v2.py','def precision_provenance():',7), location('finetune_v2.py','def provenance():',7)),
    ], [desc(WORK / MOD / 'protocol.py'), desc(OUT / 'continuation_queue_v3/registration.json')], ['Reconcile every final history LR/epoch/update count and checkpoint optimizer state against this policy.', 'Verify no outer-test artifacts were read before full360 training gate; this review performs no outer-test read.', 'Treat head-warmup selected FT states truthfully rather than replacing the fixed validation-selected checkpoint.']))
    rows.append(item('R27', 'All47 Q3 source descriptors currently match. Versioned precision, adapter, ledger and evaluator changes are explicit while prior bytes remain sealed. Input/reference/calibration/native-dimension and reproduction-scope differences must be carried into final deliverables.', [
        claim('Actual Q3 wrapper/delegate source identities are pinned.', 'Q3 validates47sources inherited fromQ2; only qualified runner mapping, owner path and E2 routes change. F2 provenance names original delegate and LunaV2, E2 names its E1 delegate/F2/actual adapter.', location('continuation_queue_v3.py','def sources():',10), location('continuation_queue_v3.py','def isolated_driver():',27), location('finetune_v2.py','def provenance():',7), location('evaluate_v2.py','def execution_provenance(',10)),
        claim('Engineering corrections preserve scientific intent and disclose exact changes.', 'SSLv2 disables loss autocast while retaining sampler/EMA/optimizer; LunaV2 excludes fixed author RoPE frequencies from trainability without changing forward math; CodeBrainV2 initializes weight-only kernel caches and excludes unused final residual parameters.', location('ssl_v2.py','def ssl_loss(output):',9), location('adapters/luna_v2.py','def exact_finetune_parameter_names(',20), location('adapters/codebrain_v2.py','def load_encoder(device=',37)),
        claim('Known methodological differences are not concealed.', 'EEGPT common-reference/uncalibrated reader scale differs from average-reference pretraining; REVE nominal metre positions and trailing80 unused samples are explicit; BIOT local5s differs from10s pretraining. Conformer microbatch BN differs from full8-person statistics; Mamba says author-public-implementation adaptation, not full paper reproduction.', location('adapters/eegpt.py','audit.update(native_dimension=',7), location('adapters/reve.py','INPUT_CONTRACT =',20), location('adapters/biot.py','INPUT_CONTRACT =',8), location('supervised.py','"batchnorm_microbatch_statistics_not_true_batch8"',1), location('adapters/eegmamba.py','INPUT_CONTRACT =',7)),
        claim('Published report logic includes source-dependent comparison scope.', 'E1 comparison contract records nativeD/head size/precision/reference/amplitude and marks Mamba scope; E2 additionally exposes exact selected/terminal encoder changes. Final actual report files have not been produced or examined here.', location('evaluate.py','def _comparison_contract(',30), location('evaluate_v2.py','def selection_evidence(',31)),
    ], [desc(OUT / 'continuation_queue_v3/registration.json'), desc(OUT / 'assets/download_manifest.json')] + engineering_desc, ['Verify final9conclusions and comparison fields report all actual adaptations/resource limitations, selectedFT phase and37-person development scope.', 'Bind final semantic companion to actual360 complete artifacts, actual evaluation complete and actual final cost artifact.', 'Any future repair must use a new version and source registration; this preflight does not authorize editing existing sealed files.']))
    rows.append(item('R29', 'Source control flow and observed handoffs/resume support a sequential single-GPU lifecycle so far. The original paused Q3 launcher later exited; its lease was archived after absence checks and full-checkpoint resume advanced. Final queue exit, Linux-child ownership and all remaining job postconditions are still pending.', [
        claim('One queue owner and one waited child control the normal path.', 'Exclusive owner creation fails on existing lease; each child is Popen then wait before next job, source hashes are rechecked and failed prerequisites block dependents. Cleanup retains lease when child termination is unverified.', location('continuation_queue.py','def acquire_owner():',18), location('continuation_queue.py','def run_child(',42), location('continuation_queue.py','def run(*, resume=False):',38)),
        claim('H2/H3 handoffs preserve child training and old source bytes.', 'Exact process birth/image/parent handles constrain parent control. Completed Conformer15 or SSL15 plus both child exit0 are required before only parent retirement; old owner/attempt/source descriptors are preserved. Guard waits for successor Popen lifetime.', location('handoff_supervisor_v2.py','def capture(self, role',25), location('handoff_supervisor_v2.py','def arm(expected',69), location('handoff_supervisor_v3.py','def arm(expected',57)),
        claim('Linux processes expose the actual process-group owner before runtime work.', '_register_owner durably records pid/pgid and confines records to legacy attempts root; Q3 rewrites only the owner-path expression into that accepted root, while its own attempt/log remainQ3.', location('linux_entry.py','def _register_owner(',22), location('continuation_queue_v3.py','def corrected_run_child(',18)),
        claim('Explicit pause and later resume preserve verifiable history.', 'Pause captured three exact process identities and suspended without termination. Recovery checked old PIDs and related experiment Python processes absent twice, copied artifacts exclusively with equalSHA, then renamed onlyQ3 owner within experiment. Old attempt/status/log bytes remain unchanged.', location('output/analysis/eeg_all_candidates_v1_20261008/pause_live_queue_v3_20261008.py','def main():',81), location('output/analysis/eeg_all_candidates_v1_20261008/prepare_q3_checkpoint_resume_20261009.py','def inspect():',10), location('output/analysis/eeg_all_candidates_v1_20261008/prepare_q3_checkpoint_resume_20261009.py','def preserve(',10), location('output/analysis/eeg_all_candidates_v1_20261008/prepare_q3_checkpoint_resume_20261009.py','observed = inspect()',9)),
        claim('Checkpoint replay cost remains distinct from durable formal work.', 'Saved epoch26/104updates/702backward versus acknowledged epoch27/108/729 yields exactly4 updates and27 backward to repeat. Diagnostic forwards/time remainunknown. Recorded resume reached epoch49/196updates and proves original resume path advanced beyondpause. No suspension interval is counted as training.', location('supervised.py','def validate_resume(',25), location('supervised.py','target = folder / "latest.pt"',17), location('output/analysis/eeg_all_candidates_v1_20261008/prepare_q3_checkpoint_resume_20261009.py',"'acknowledged_updates_not_in_checkpoint': 4",8)),
    ], lifecycle + [desc(pause_path), desc(resume_path), desc(advanced_path), desc(h3_state_path), resume['owner_archived'], archived_old_status, archived_old_log], ['At final completion inspectQ3 terminal status, each actual attempt returncode/postcondition, absence of active owned GPU children and release of the current owner.', 'Require each Linux action owner/runtime/attempt binding; Windows launcher exit alone is insufficient for orphan safety.', 'Keep known original H3 successor launcher exit distinct from unknown paused GPU/dispatcher child exit codes.']))
    statuses = {}
    for fit in registration['fits']:
        path = OUT / 'runs' / fit['condition'] / ('seed_' + str(fit['seed'])) / ('fold_' + str(fit['fold'])) / 'status.json'
        status = read(path)['status'] if path.exists() else 'engineering_pending'
        statuses[status] = statuses.get(status, 0) + 1
    result = {'kind': 'independent_semantic_preflight_not_final', 'schema_version': 1,
              'reviewer': '/root/resume_review', 'preparation_complete': True, 'full_plan_completed': False,
              'review_started_at': started, 'review_closed_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'source': desc(__file__), 'registration': reg_desc, 'q3_registration': desc(OUT / 'continuation_queue_v3/registration.json'),
              'q3_source_descriptors_checked': q3['sources'], 'q3_source_count': 47, 'status_snapshot': statuses,
              'scope': {'static_source_interpretation': True, 'json_log_and_artifact_byte_hash_checks': True,
                        'model_constructions': 0, 'tensor_checkpoint_loads': 0, 'raw_signal_reads': 0,
                        'feature_array_reads': 0, 'outer_test_probability_or_metric_reads': 0,
                        'gpu_calls': 0, 'optimizer_updates': 0, 'process_actions': 0,
                        'sealed_sources_or_old_artifacts_modified': False},
              'source_repository_bindings_from_download_manifest': sources, 'engineering_snapshot': engineering,
              'items': rows,
              'historical_resource_measurement_gaps_and_future_supplement_plan': {
                  'historical_unknowns_must_remain_unknown': True, 'encoder_export_peak_is_not_head_probe_peak': True,
                  'formal_fit_resources_are_real_but_not_old_probe_measurements': True,
                  'listed_current_conditions': supplemental,
                  'future_mamba_frozen_same_code_gap_expected': True,
                  'execution_not_performed_or_scheduled_by_this_review': True,
                  'preconditions': ['Actual formal/evaluation GPUqueue fully ended and no owned GPU child remains.',
                                    'Re-enumerate all24 actual engineering locks after final training; pin their descriptors.',
                                    'Use the same sealed head/feature manifests and exact fold1 train IDs; neverread raw EEG/testprobabilities.'],
                  'minimum_new_measurement': ['Fresh seeded head, exact recorded CUDA float32 inputs and native dimension; one train-only forward/backward, finite-grad check, zerooptimizersteps.',
                                              'Roundtrip model/empty optimizer/RNG state as the original engineering probe; use a new independent output path.',
                                              'Create attempt event before data/model work; measure synchronized CUDA elapsed and separate whole-attempt elapsed, reset/measure allocated+reserved peaks with baseline.',
                                              'Record actual started/returned forward/backward/step counts, source/manifest/checkpoint hashes, trainIDs and device/runtime.',
                                              'Report new measurement only as supplementary qualification; old seconds/peaks remain unknown and all new cost is additional.']},
              'pause_resume_review': {'pause': desc(pause_path), 'recovery': desc(resume_path), 'observed_advanced': desc(advanced_path),
                                    'archived_owner_matches_pause_bytes': True, 'preserved_artifact_pairs_verified': len(resume['preserved_artifacts']),
                                    'old_attempt_status_and_log_still_match_archive': True,
                                    'uncheckpointed_acknowledged_optimizer_updates': 4, 'uncheckpointed_acknowledged_backward_calls': 27,
                                    'uncheckpointed_diagnostic_forwards': None, 'prior_partial_model_seconds': None,
                                    'paused_three_process_exit_codes_or_times': None,
                                    'separate_original_h3_successor_launcher_observation': {k: h3_state.get(k) for k in ('v3_launcher_pid','v3_exit_code','v3_finished_at','phase')},
                                    'h3_launcher_exit_does_not_infer_each_paused_child_exit': True},
              'final_companion_requirements': 'This is not kind=independent_final_semantic_review and contains no passed=true completion assertion. After actual360/435 evaluation and final cost exist, rereview pending findings and bind final companion to full final evidence using audit_full_plan_acceptance_v3 semantic_review_binding.'}
    for record in q3['sources']:
        check(record)
    target = REPORT / ('semantic_preflight_' + str(time.time_ns()) + '.json')
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'artifact': desc(target), 'items': len(rows), 'sources': 47,
                      'current_engineering_locks': len(engineering), 'historical_probe_resource_gap_conditions': len(supplemental),
                      'full_plan_completed': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
