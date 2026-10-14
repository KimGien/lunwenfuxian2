"""Stdlib-only Linux admission identities, with no model or GPU operations.

The Windows controller independently captures one exact module/nonce worker.
That worker must see the same boot, pid, parent, start ticks, argv, executable
and parent identity before model work and at every subsequent owned boundary.
"""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

SOURCE=Path(__file__).resolve()
WORK=SOURCE.parents[2]
MODULE_PATTERN=r'eeg_all_candidates|audit_completed_finetune|qualify_.*sync|sync_qualification_boundary|supplemental_head_resource'

def require(ok,message):
    if not ok:raise ValueError(message)

def process_identity(pid,proc_root=Path('/proc')):
    folder=proc_root/str(pid)
    fields=(folder/'stat').read_text().rsplit(')',1)[1].split()
    argv=[x.decode('utf-8',errors='strict') for x in (folder/'cmdline').read_bytes().split(b'\0') if x]
    return {'pid':int(pid),'parent':int(fields[1]),'start_ticks':fields[19],
            'argv':argv,'executable':str((folder/'exe').resolve(strict=True))}

def snapshot_local(*,exclude_pid=None,proc_root=Path('/proc')):
    require(sys.platform=='linux','Local process identity requires Linux')
    rows=[];errors=[]
    for folder in proc_root.iterdir():
        if not folder.name.isdigit() or int(folder.name)==exclude_pid:continue
        try:
            command=(folder/'cmdline').read_bytes().replace(b'\0',b' ').decode('utf-8',errors='strict')
            if not re.search(MODULE_PATTERN,command):continue
            row=process_identity(int(folder.name),proc_root)
            row['parent_identity']=process_identity(row['parent'],proc_root)
            rows.append(row)
        except FileNotFoundError:continue
        except (OSError,ValueError,UnicodeError) as error:
            errors.append({'pid':int(folder.name),'error_type':type(error).__name__})
    return {'schema':'linux_worker_identity_snapshot_v1','query_succeeded':True,
            'boot_id':(proc_root/'sys/kernel/random/boot_id').read_text().strip(),
            'rows':sorted(rows,key=lambda x:x['pid']),'errors':errors}

def linux_path(path):
    path=Path(path).resolve()
    require(path.drive and len(path.drive)==2,'Windows drive required')
    return '/mnt/'+path.drive[0].lower()+'/'+path.as_posix()[3:]

def snapshot():
    if sys.platform=='linux':return snapshot_local()
    require(sys.platform=='win32','Only pinned Windows/WSL bridge supported')
    # The extra stdlib reader excludes itself from the experimental inventory.
    # It constructs no model and imports no Torch, Mamba, adapter or raw data.
    source_path=linux_path(SOURCE)
    program=("import importlib.util,json,os; s=importlib.util.spec_from_file_location('_linux_admission_reader',"+
             repr(source_path)+"); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
             "print(json.dumps(m.snapshot_local(exclude_pid=os.getpid())))")
    command=['wsl.exe','-d','TRI-DEP-Ubuntu-24.04','--exec',
             linux_path(WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/assets/eegmamba_linux_env/bin/python'),
             '-B','-c',program]
    result=subprocess.run(command,capture_output=True,text=True,check=True,timeout=30)
    return json.loads(result.stdout.strip().splitlines()[-1])

def select_unique(value,*,module,flags):
    require(value.get('query_succeeded') is True and value.get('errors')==[]
            and type(value.get('boot_id')) is str and bool(value['boot_id']),
            'Linux inventory unavailable or incomplete')
    rows=value['rows']
    require(type(rows) is list and len(rows)==1,'Exactly one experimental Linux worker required')
    row=rows[0];argv=row['argv']
    require(type(row['pid']) is int and row['pid']>0 and type(row['parent']) is int and row['parent']>0
            and type(row['start_ticks']) is str and row['start_ticks'].isdigit(), 'Exact Linux pid/parent/start ticks required')
    require(argv.count('-m')==1 and argv[argv.index('-m')+1]==module,
            'Linux worker module differs from controller launch')
    for flag,expected in flags.items():
        require(argv.count(flag)==1 and argv[argv.index(flag)+1]==str(expected),
                'Linux worker nonce/admission/role differs: '+flag)
    parent=row.get('parent_identity')
    require(type(parent) is dict and parent['pid']==row['parent'] and parent['start_ticks'].isdigit()
            and parent['argv'] and parent['executable'],'Complete Linux parent identity required')
    require(row['executable'] and argv[0],'Linux executable/argv absent')
    return {'boot_id':value['boot_id'],'identity':row}

def validate_bound(value,binding,*,module,flags,worker_pid):
    actual=select_unique(value,module=module,flags=flags)
    require(type(worker_pid) is int and actual['identity']['pid']==worker_pid,
            'Actual Linux caller is not controller-captured worker')
    require(actual==binding,'Linux boot/pid/parent/birth/argv/executable identity changed')
    return actual

def wait_capture(*,module,flags,is_alive,timeout=20):
    deadline=time.monotonic()+timeout
    while True:
        require(is_alive() and time.monotonic()<deadline,'Linux child exited or identity handshake timed out')
        value=snapshot()
        require(value.get('query_succeeded') is True and value.get('errors')==[], 'Linux inventory query failed')
        if value['rows']:return select_unique(value,module=module,flags=flags)
        time.sleep(.1)
