"""Read-only verification of the sealed 128-dimensional source cache graph."""
from __future__ import annotations
import hashlib
import ast
import time
import numpy as np
from . import common as C

def descriptors(value, found):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            old = found.get(value['path'])
            C.require(old is None or old == value['sha256'], 'Conflicting source binding: ' + value['path'])
            found[value['path']] = value['sha256']
        for v in value.values():
            descriptors(v, found)
    elif isinstance(value, list):
        for v in value:
            descriptors(v, found)

def run():
    C.check_pause()
    destination = C.OUT/'source_gate.json'
    if destination.exists():
        saved = C.read(destination)
        C.require(saved['passed'], 'Prior source gate did not pass')
        for record in saved['small_bindings']:
            C.checked(record)
        return saved
    started = time.perf_counter()
    old_gate = C.read(C.V2_OUT/'feature_gate.json')
    C.require(old_gate['passed'] and old_gate['bundles'] == 240, 'Sealed feature gate incomplete')
    found = {}
    # Feature metadata binds parent checkpoint, original inputs, extraction code,
    # subjects, roles and the pre-N1 classifier replay, rather than old fit scores.
    for item in old_gate['sources']:
        C.checked(item)
        descriptors(item, found)
    caches = []
    cohort = None
    by_fold = {}
    for seed in C.SEEDS:
        for fold in range(1,6):
            st_meta = C.read(C.V2_OUT/'features/ST'/f'seed_{seed}'/f'fold_{fold}.json')
            for eeg in ('ST', *C.EEG):
                C.check_pause()
                path = C.V2_OUT/'features'/eeg/f'seed_{seed}'/f'fold_{fold}.npz'
                meta_path = path.with_suffix('.json')
                meta = C.read(meta_path)
                descriptors(meta, found)
                roles = meta['roles']
                C.require(roles == st_meta['roles'], 'E/ST frozen roles differ')
                role_people = sum([roles[k] for k in ('train','val','test')], [])
                C.require(len(role_people)==37 and len(set(role_people))==37 and len(roles['val'])==3, 'Invalid frozen split')
                with np.load(path, allow_pickle=False) as z:
                    ids, labels, splits, x = [z[k] for k in ('subject_ids','labels','splits','features')]
                    C.require(len(set(ids.tolist()))==37 and set(ids.tolist())==set(role_people), 'Cohort or cache identity differs')
                    C.require(set(labels.tolist())=={0,1} and int(labels.sum())==16, 'Cohort labels differ')
                    C.require(x.shape == ((37,2,128) if eeg=='ST' else (37,128)), '128D interface differs')
                    C.require(x.dtype==np.float32 and np.isfinite(x).all(), 'Cache dtype/finiteness invalid')
                    current = dict(zip(ids.tolist(),labels.tolist()))
                    if cohort is None: cohort = current
                    C.require(cohort==current, 'People or labels differ across source identities')
                    C.require(all(str(splits[i]) in roles and str(sid) in roles[str(splits[i])] for i,sid in enumerate(ids)), 'Role labels differ')
                if eeg in ('ST','E00'):
                    replay = meta['final_linear_replay']
                    C.require(len(replay)==3 and all(r['max_probability_error']<=r['atol'] for r in replay), 'Stage0 classifier replay evidence invalid')
                else:
                    C.require(meta['replay_passed'] and meta['parent_state_unchanged'] and meta['base_model_updates']==0 and meta['subjects_forwarded']==37, 'EEG frozen-source gate invalid')
                    C.require(meta['max_logit_error']<=meta['replay_atol'] and meta['max_reconnect_error']<=meta['replay_atol'], 'EEG classifier replay tolerance exceeded')
                    parent = meta['source_unit']['checkpoint']['path']
                    if eeg=='E15':
                        C.require('LaBraM-native4s-originalCNN-300ep' in parent and parent.endswith('/epoch_300.pt'), 'E15 endpoint changed')
                    else:
                        C.require(parent.endswith('/best.pt'), 'EEG sealed best changed')
                caches.append({'eeg':eeg,'seed':seed,'fold':fold,'cache':C.desc(path),'metadata':C.desc(meta_path),
                               'pre_N1_classifier_replay_reused':True})
            by_fold[(seed,fold)] = roles
    for seed in C.SEEDS:
        test_ids = sum([by_fold[(seed,f)]['test'] for f in range(1,6)],[])
        C.require(len(test_ids)==37 and len(set(test_ids))==37, 'Outer test coverage differs')
    verified = []
    total_bytes = 0
    for i,(path,digest) in enumerate(sorted(found.items())):
        C.check_pause()
        record = {'path':path,'sha256':digest}
        p = C.checked(record)
        total_bytes += p.stat().st_size
        verified.append(record)
        if (i+1)%200==0:
            print('Source hashes verified',i+1,'/',len(found),flush=True)
    # Verify the pinned upstream snapshots using Git blob identities, not filenames.
    author_path = C.ROOT/'tridep_ccml/fusion_eeg_grid_v2/upstream_sources.json'
    authors = C.read(author_path)
    author_records = []
    for item in authors:
        p = C.ROOT/item['local_path']
        b = p.read_bytes()
        original = C.REPORT/'author_originals'/item['method']/item['path']
        raw = original.read_bytes()
        C.require(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==item['git_blob'], 'Original Git blob differs')
        normalize = lambda data: data.decode('utf-8').replace('\r\n','\n').rstrip('\n')
        C.require(normalize(b)==normalize(raw) and ast.dump(ast.parse(b))==ast.dump(ast.parse(raw)), 'Local vendor has substantive changes')
        author_records.append(dict(item, local_sha256=C.sha(p), original=C.desc(original),
                                   byte_difference='LF conversion and terminal newlines only; original Git blob verified'))
    small = [C.desc(C.PLAN), C.desc(C.V2_OUT/'feature_gate.json'), C.desc(C.V2_OUT/'source_registration.json'),
             C.desc(author_path), C.desc(__file__), C.desc(C.REPORT/'author_source_downloads.json'),
             C.desc(C.REPORT/'author_source_comparison.json')]
    small.extend(r['original'] for r in author_records)
    result = {'schema':'fusion_v3_source_gate_v1','passed':True,'caches':caches,'bundles':240,
              'subjects_unique':37,'MDD':16,'HC':21,'all_frozen_folds_match':True,
              'source_descriptors':verified,'source_count':len(verified),'bytes_hashed':total_bytes,
              'small_bindings':small,'authors':author_records,
              'original_replay_reused_by_exact_metadata_and_parent_hash':True,
              'new_feature_exports':0,'new_backbone_forwards':0,'new_optimizer_updates':0,
              'source_verification_seconds':time.perf_counter()-started,'at':C.now(),
              'scope':'source/interface evidence only; old QMF objective gate is NOT accepted'}
    C.save(destination,result,immutable=True)
    print('Source gate passed:',len(verified),'hashes;',len(caches),'bundles',flush=True)
    return result

if __name__=='__main__':
    run()

