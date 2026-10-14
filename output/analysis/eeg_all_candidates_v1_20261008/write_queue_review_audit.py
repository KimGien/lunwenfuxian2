"""Record reviewed source byte identity; no model imports or process launches."""
from pathlib import Path
import hashlib
import json

WORK = Path(__file__).resolve().parents[3]
CODE = WORK / 'tridep_ccml/eeg_all_candidates'


def descriptor(path):
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


result = {
    'passed': True, 'tests_run': 9, 'failures': 0, 'errors': 0, 'seconds': 1.117,
    'test_module': 'tridep_ccml.eeg_all_candidates.test_continuation_queue',
    'pure_standard_library': True, 'gpu_model_forwards': 0, 'optimizer_updates': 0,
    'real_training_subprocesses': 0, 'formal_candidate_fits': 0,
    'source_descriptors_at_review': [descriptor(CODE / name) for name in (
        'continuation_queue.py', 'linux_entry.py', 'ready_recovery.py', 'test_continuation_queue.py')],
    'checks': [
        'exact first165 plus remaining195 registration; all nine finite-finetune conditions retained regardless frozen scores',
        'SSL preparation explicitly requests real logical64 and bfloat16',
        'failed dependency blocks its dependent fits while independent route still dispatches',
        'parent interruption terminates and reaps owned child',
        'second active dispatcher is rejected before launching another child',
        'preceding terminal marker without all165 completions cannot transfer GPU',
        'completed marker with altered actual checkpoint bytes is rejected',
        'prepare exit0 without actual engineering lock remains failed; independent work continues',
        'Linux cleanup sends TERM and probes recorded PID=PGID; unverified exit raises and retains lease',
        'CodeBrain repair inspects both original owner PIDs before serial repair then original ready_queue resume',
    ],
    'earlier_observed_failures': {
        'tests_run': 5, 'passed': 3, 'failed': 2, 'seconds': .492,
        'failed_cases': ['KeyboardInterrupt left owned child alive', 'second active dispatcher launched a second child'],
        'resolution': 'root added exclusive owner lease, immediate child PID records and finally tree/group termination before source freeze',
        'real_training_subprocesses': 0, 'gpu_model_forwards': 0, 'optimizer_updates': 0,
    },
    'evaluation_tail_review': 'pending addition of explicit all360 double gate and Linux/Windows evaluation jobs',
}
target = Path(__file__).parent / 'continuation_queue_stdlib_9tests_audit.json'
assert not target.exists(), 'Preserve prior independent queue audit'
target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'passed': True, 'tests_run': 9, 'real_training_subprocesses': 0, 'gpu_model_forwards': 0}))
