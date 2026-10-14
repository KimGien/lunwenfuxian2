"""Release only the byte-bound Oct11 pause controls after explicit Oct12 continue."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

SOURCE=Path(__file__).resolve();WORK=SOURCE.parents[3]
OUT=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
def desc(p):
    p=Path(p).resolve()
    return {'path':p.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,v):
    with p.open('x',encoding='utf-8') as f:
        json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--source-review',type=Path)
    a=ap.parse_args()
    if not a.run:
        print(json.dumps({'status':'pending_explicit_continue_release','process_actions':0}));return
    assert a.source_review is not None
    review=json.loads(a.source_review.read_text('utf-8-sig'))
    assert review['schema']=='q5_recovery_v3_independent_source_review' and review['passed'] is True
    assert len(review['sources'])==88 and all(desc(WORK/x['path'])==x for x in review['sources'])
    pointer=OUT/'user_pause_q5_owner_20261011.json'
    intent=json.loads(pointer.read_text('utf-8-sig'))
    pause=OUT/'pause_events/q5_user_pause_1791729696123'
    assert Path(intent['folder'])==pause and intent['automatic_resume_authorized'] is False
    assert not (pause/'resume_intent.json').exists()
    native=pause/'native_pause_complete.json'
    assert desc(native)['sha256']=='65977f9551a7621f9df0e1701173bf1845794a6c60b7a0d58012237092a98c40'
    paths=[pointer]
    for name in ('sync_boundary','sync_boundary_v2','sync_boundary_v3'):
        p=OUT/'execution_acceleration_v1'/name/'user_pause.json'
        m=json.loads(p.read_text('utf-8-sig'))
        assert set(m)=={'paused'} and m['paused'] is True
        paths.append(p)
    folder=OUT/'process_loss_recovery_v3'/('explicit_continue_'+str(time.time_ns()))
    folder.mkdir()
    write(pause/'disk_recovery_continue_intent.json',{'source':desc(SOURCE),'explicit_user_continue_date':'2026-10-12','resume_native_calls':0,'folder':str(folder),'retry_permitted':False})
    retained=[]
    for i,p in enumerate(paths):
        original=desc(p);target=folder/f'control_{i}.json'
        with target.open('xb') as f:
            f.write(p.read_bytes());f.flush();os.fsync(f.fileno())
        assert desc(target)['sha256']==original['sha256'] and desc(p)==original
        retained.append({'original':original,'snapshot':desc(target)})
    write(folder/'authorization.json',{'source':desc(SOURCE),'explicit_user_continue_date':'2026-10-12','independent_source_review':desc(a.source_review),'old_pause_proof':desc(native),'controls':retained,'process_actions':0,'model_operations':0,'training_started':False,'old_process_endpoint_unknown':True,'overnight_pause_not_training':True,'exact_active_elapsed_seconds':None})
    for row in retained:
        p=WORK/row['original']['path']
        assert desc(p)==row['original']
        p.unlink()
    write(folder/'controls_released.json',{'source':desc(SOURCE),'authorization':desc(folder/'authorization.json'),'released_controls':retained,'process_actions':0,'model_operations':0})
    print(json.dumps({'artifact':desc(folder/'controls_released.json'),'folder':str(folder),'status':'explicit_continue_archived_controls_released'}))
if __name__=='__main__':main()
