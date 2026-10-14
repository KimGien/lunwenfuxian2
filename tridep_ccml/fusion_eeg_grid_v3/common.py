from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import time
import uuid
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OUT = ROOT / 'tridep_ccml/experiments/Fusion-EEG15-Methods5-v3-37-20261014'
V2_OUT = ROOT / 'tridep_ccml/experiments/Fusion-EEG15-Methods5-v2-37-20261013'
REPORT = ROOT / 'output/analysis/fusion_eeg15_methods5_v3_20261014'
PLAN = ROOT / 'output/plans/EEG十五版本与五种融合方法后续计划_v3_20261014.md'
OLD = ROOT / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
STAGE0 = ROOT / 'output/analysis/tridep_stage0_20261004'
SEEDS = (11, 22, 33)
METHODS = ('CCML', 'LMF', 'MMDynamics', 'QMF', 'ECML')
EEG = {'E00': 'original-CBraMod-CNN', 'E01': 'EEGPT-frozen-smallhead',
       'E02': 'CSBrain-frozen-smallhead', 'E03': 'REVE-frozen-smallhead',
       'E04': 'LUNA-frozen-smallhead', 'E05': 'CodeBrain-frozen-smallhead',
       'E06': 'BIOT-frozen-smallhead', 'E07': 'EEGMamba-frozen-smallhead',
       'E08': 'EEGConformer-scratch', 'E09': 'EEG2Rep-supervised',
       'E10': 'LaBraM-native4s-frozen-smallhead', 'E11': 'LaBraM-native5s-frozen-smallhead',
       'E12': 'LaBraM-native4s-lastblock-finetune', 'E13': 'LaBraM-native5s-lastblock-finetune',
       'E15': 'LaBraM-native4s-originalCNN-300ep'}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def desc(path):
    return {'path': Path(path).resolve().relative_to(ROOT).as_posix(), 'sha256': sha(path)}

def checked(record):
    path = (ROOT / record['path']).resolve()
    require(path.is_relative_to(ROOT) and sha(path) == record['sha256'], 'Source changed: ' + str(path))
    return path

def atomic(path, data, immutable=False):
    path = Path(path).resolve()
    require(any(path.is_relative_to(p.resolve()) for p in (OUT, REPORT)), 'Output outside new experiment')
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        require(path.read_bytes() == data, 'Immutable artifact differs: ' + str(path))
        return
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.part')
    with temp.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(40):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.1)

def save(path, value, immutable=False):
    atomic(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode(), immutable)

def save_torch(path, value):
    import torch
    stream = io.BytesIO()
    torch.save(value, stream)
    atomic(path, stream.getvalue())

def save_npz(path, **arrays):
    import numpy as np
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    atomic(path, stream.getvalue())

def paused():
    p = OUT / 'control/user_pause.json'
    require(p.exists(), 'Missing v3 user pause control; inspect before creating it')
    v = read(p)
    require(set(v) == {'paused'} and isinstance(v['paused'], bool), 'Invalid user pause marker')
    return v['paused']

def check_pause():
    if paused():
        raise InterruptedError('User paused experiment at durable boundary')

def inventory():
    # Query the actual native tree, including venv launcher and its real child.
    script = "$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|wsl' -and $_.CommandLine -match 'lunwen2|tridep_ccml' } | ForEach-Object { $p=Get-Process -Id $_.ProcessId -ErrorAction Stop; [pscustomobject]@{pid=$_.ProcessId;parent=$_.ParentProcessId;birth=$p.StartTime.ToUniversalTime().ToFileTimeUtc().ToString();command=$_.CommandLine;executable=$_.ExecutablePath} }) | ConvertTo-Json -Depth 5 -Compress"
    import base64
    r = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-EncodedCommand',
                        base64.b64encode(script.encode('utf-16le')).decode()],
                       capture_output=True, text=True, check=True, timeout=30)
    data = json.loads(r.stdout.lstrip('\ufeff')) if r.stdout.strip() else []
    return [data] if isinstance(data, dict) else data

def own_tree():
    data = inventory()
    own = next((x for x in data if x['pid'] == os.getpid()), None)
    require(own is not None, 'Native process birth identity unavailable')
    found = [own]
    p = next((x for x in data if x['pid'] == own['parent']), None)
    if p and p['command'] == own['command']:
        found.append(p)
    return found

def runtime(seed, device='cpu'):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == 'cuda':
        require(torch.cuda.is_available(), 'CUDA unavailable')

def cpu_state(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_state(x) for x in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(x) for x in value)
    return value

def state_sha(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        x = v.detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()

def rng_state():
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(value):
    import numpy as np
    import torch
    random.setstate(value['python']); np.random.set_state(value['numpy']); torch.set_rng_state(value['torch'])
    if value['cuda']:
        torch.cuda.set_rng_state_all(value['cuda'])

def identities():
    for e in ('ST', *EEG):
        for method in METHODS:
            for seed in SEEDS:
                for fold in range(1, 6):
                    yield {'eeg': e, 'method': method, 'seed': seed, 'fold': fold}

def fitdir(identity):
    stage = identity.get('stage', 'B_main')
    require(stage in ('B_main', 'B_N1Q0', 'B_N0Q1', 'C_panel_lr3e-4', 'C_panel_lr3e-3', 'C_transfer', 'D_extension'), 'Unknown experiment stage')
    require(identity['eeg'] in ('ST', *EEG) and identity['method'] in METHODS and identity['seed'] in SEEDS and identity['fold'] in range(1, 6), 'Unknown formal identity')
    return OUT / 'runs' / stage / (identity['eeg'] + '-' + identity['method']) / f"seed_{identity['seed']}" / f"fold_{identity['fold']}"
