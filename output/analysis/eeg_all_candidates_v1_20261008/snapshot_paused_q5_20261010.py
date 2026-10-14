"""Archive bytes of an already verified paused Q5; never resume or decode tensors."""
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from tridep_ccml.eeg_all_candidates import storage as S, registry as R

SOURCE = Path(__file__).resolve()
FOLDER = S.OUT/'pause_events/q5_user_pause_1791607365360'
native = S.read(FOLDER/'native_pause_complete.json')
assert native['status'] == 'paused_by_explicit_user'
assert len(native['actions']) == 4 and all(row['native_status'] == 0 for row in native['actions'])
assert all(t['wait_reason'] == 'Suspended' for row in native['verification'] for t in row['threads'])
active = []
for path in (S.OUT/'runs/REVE-lastblock-finetune').glob('seed_*/fold_*/status.json'):
    value = S.read(path)
    if value['status'] == 'training': active.append((path.parent,value))
assert len(active) == 1
fit,status = active[0]
copies=[]
def archive(path, name):
    if not path.exists(): return
    original=S.desc(path); target=FOLDER/name
    if target.exists(): raise ValueError('Pause copy already exists; inspect rather than overwrite')
    shutil.copy2(path,target)
    assert S.desc(path) == original and S.digest(target) == original['sha256']
    copies.append({'original':original,'snapshot':S.desc(target)})
for name in ('initial.pt','latest.pt','best.pt','train_history.json','validation_history.json',
             'progress.json','started.json','status.json','initial_metadata.json'):
    archive(fit/name,name)
owner=S.read(S.OUT/'continuation_queue_v5/owner.json')
assert owner == native['owner_snapshot']
archive(S.OUT/'continuation_queue_v5/owner.json','q5_owner.json')
archive((S.WORK/owner['linux_owner']).with_name('status.json'),'q5_attempt_status.json')
attempts=[]
for path in (S.OUT/'finetune_v2_attempts/REVE-lastblock-finetune').glob('*/attempt.json'):
    value=S.read(path)
    if value.get('kind') == 'formal' and value.get('seed') == status['seed'] and value.get('fold') == status['fold'] and value.get('status') == 'running':
        original=S.desc(path); archive(path,'buffered_attempt.json')
        attempts.append({'original':original,'snapshot':S.desc(FOLDER/'buffered_attempt.json'),
                         'unflushed_tail_unknown_if_runtime_lost':True})
assert len(attempts) == 1
proof={'status':'paused_by_user','source':S.desc(SOURCE),'native_pause':S.desc(FOLDER/'native_pause_complete.json'),
       'native_control_source':native['source'],'paused_at':native['paused_at'],'snapshot_completed_at':S.now(),
       'owner_snapshot':owner,'identities':native['identities'],'actions':native['actions'],
       'condition':status['condition'],'seed':status['seed'],'fold':status['fold'],
       'active_progress':S.read(fit/'progress.json'),'retained':copies,'attempt_snapshots':attempts,
       'registry_status':R.status_summary(),'process_terminate_calls':0,'tensor_deserializations':0,
       'model_constructions':0,'optimizer_operations':0,'GPU_model_operations':0,
       'checkpoint_byte_copies_and_hash_reads_performed':True,'runtime_memory_currently_preserved':True,
       'pause_interval_not_training':True,'automatic_resume_authorized':False,
       'resume_note':'Only after explicit human continue. Verify every exact native birth and partial suspend record. All four objects were suspended once; do not blindly repeat resume, restart old queues, or clear controls. If runtime was lost, preserve old bytes and unknown buffered tail/time, validate full CPU checkpoint and create a new reviewed recovery version.'}
S.save(FOLDER/'pause_complete.json',proof,immutable=True)
import json
print(json.dumps({'artifact':S.desc(FOLDER/'pause_complete.json'),'registry':proof['registry_status'],
                  'active':[proof['condition'],proof['seed'],proof['fold']],'progress':proof['active_progress']}))
