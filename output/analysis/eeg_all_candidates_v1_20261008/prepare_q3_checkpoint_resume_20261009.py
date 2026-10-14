"""Archive a proven-dead Q3 lease and preserve the paused checkpoint/cost evidence."""
from pathlib import Path
import json
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import registry as R, storage as S


def require(value, message):
    if not value:
        raise ValueError(message)


def inspect():
    script = ("Get-CimInstance Win32_Process | Where-Object { "
              "($_.ProcessId -in @(39760,46676,21104,39768,39180)) -or "
              "(($_.Name -match '^python(w)?\\.exe$') -and "
              "($_.CommandLine -match 'tridep_ccml\\.eeg_all_candidates\\.')) } "
              "| Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine "
              "| ConvertTo-Json -Depth 3")
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', script], capture_output=True,
                            text=True, encoding='utf-8', check=True)
    return json.loads(result.stdout) if result.stdout.strip() else []


def main():
    pause_path = S.OUT / 'pause_events/1791460680736346200/pause_complete.json'
    pause = S.read(pause_path)
    require(pause['status'] == 'paused_by_user', 'Expected explicit recorded pause')
    owner_path = S.checked(pause['owner'])
    require(S.read(owner_path) == pause['owner_value'], 'Prior owner changed')
    require(inspect() == [], 'Old or related experiment process still exists; do not archive lease')
    token = str(time.time_ns())
    archive = (S.OUT / 'resume_events' / token).resolve()
    require(archive.is_relative_to(S.OUT.resolve()) and owner_path.resolve().is_relative_to(S.OUT.resolve()),
            'Archive/source must stay in this experiment')
    archive.mkdir(parents=True, exist_ok=False)
    archived = []

    def preserve(path, name):
        path = Path(path)
        before = S.desc(path)
        target = archive / name
        with target.open('xb') as stream:
            stream.write(path.read_bytes())
        after = S.desc(target)
        require(before['sha256'] == after['sha256'], 'Archived bytes differ')
        archived.append({'original_at_resume': before, 'preserved_copy': after})
        return after

    fit = R.fit_directory('EEG2Rep-supervised', 22, 4)
    active = pause['active_fit_progress'][0]
    S.checked(active['progress'])
    S.checked(active['latest_checkpoint'])
    history = S.read(fit / 'train_history.json')
    progress = S.read(fit / 'progress.json')
    require(history[-1]['epoch'] == 26 and history[-1]['optimizer_updates'] == 104
            and history[-1]['backward_calls'] == 702, 'Durable training history differs')
    require(progress == active['value'] and progress['epoch'] == 27
            and progress['optimizer_updates_completed'] == 108
            and progress['backward_calls_completed'] == 729, 'Paused acknowledged cost differs')
    for name in ('latest.pt', 'progress.json', 'train_history.json', 'validation_history.json',
                 'started.json', 'status.json'):
        preserve(fit / name, 'paused_fit_' + name)
    old_attempt = S.OUT / 'continuation_queue_v3/attempts/1791458796212822000-eeg2rep-supervised-fit'
    preserve(old_attempt / 'status.json', 'old_attempt_status.json')
    preserve(old_attempt / 'process.log', 'old_attempt_process.log')
    preserve(S.OUT / 'continuation_queue_v3/status.json', 'old_queue_status.json')
    observed = inspect()
    require(observed == [] and S.desc(owner_path) == pause['owner'], 'Process/owner changed before lease archive')
    target = archive / 'q3_owner_prior.json'
    require(target.resolve().is_relative_to(S.OUT.resolve()) and not target.exists(), 'Unsafe archive target')
    owner_path.rename(target)
    owner_archive = S.desc(target)
    require(owner_archive['sha256'] == pause['owner']['sha256'], 'Archived lease bytes differ')
    event = {'kind': 'user_authorized_resume_after_paused_process_exit', 'recorded_at': S.now(),
             'status': 'checkpoint_resume_prepared', 'source': S.desc(Path(__file__)),
             'user_authorization': '2026-10-09 explicit continue; supersedes prior pause',
             'pause_record': S.desc(pause_path), 'prior_owner': pause['owner'], 'owner_archived': owner_archive,
             'old_exact_pids_and_related_experiment_python_processes_observed': observed,
             'old_process_exit_code': None, 'old_process_exit_time': None,
             'original_attempt_status_or_logs_modified': False, 'sealed_sources_modified': False,
             'condition': 'EEG2Rep-supervised', 'seed': 22, 'fold': 4,
             'durable_epoch': 26, 'durable_optimizer_updates': 104, 'durable_backward_calls': 702,
             'paused_epoch': 27, 'paused_acknowledged_optimizer_updates': 108,
             'paused_acknowledged_backward_calls': 729,
             'acknowledged_updates_not_in_checkpoint': 4,
             'acknowledged_backward_calls_not_in_checkpoint': 27,
             'uncheckpointed_diagnostic_forward_count': None,
             'prior_partial_attempt_model_seconds': None,
             'cost_semantics': 'Four acknowledged optimizer updates and27 backward calls occurred after the retained '
                               'checkpoint and will be repeated during normal epoch27 recovery. Keep this prior work '
                               'in addition to durable formal counts; unrecorded diagnostics/timing remain unknown. '
                               'Overnight suspension is not training time.',
             'checkpoint_cpu_preflight': {'epoch': 26, 'optimizer_updates': 104, 'backward_calls': 702,
                                         'history_and_orders': 26, 'adamw_states_at_step104': 88,
                                         'rng_and_sampler_present': True, 'all_state_tensors_finite': True,
                                         'model_constructions': 0, 'gpu_calls': 0},
             'preserved_artifacts': archived, 'registry_status': R.status_summary(),
             'next_command': ['S00/.venv/Scripts/python.exe', '-B', '-u', '-m',
                              'tridep_ccml.eeg_all_candidates.continuation_queue_v3', '--resume']}
    path = S.OUT / 'resume_events' / (token + '.json')
    S.save(path, event, immutable=True)
    print(json.dumps({'prepared': True, 'event': S.desc(path), 'archived_lease': owner_archive,
                      'acknowledged_updates_to_repeat': 4, 'acknowledged_backward_calls_to_repeat': 27}))


if __name__ == '__main__':
    main()
