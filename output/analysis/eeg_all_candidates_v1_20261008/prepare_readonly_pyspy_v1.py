"""Stage a hash-verified official profiler binary; do not change training env."""
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
from urllib.parse import urlparse
import zipfile

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import storage as S


def run():
    begin = time.perf_counter()
    with urllib.request.urlopen('https://pypi.org/pypi/py-spy/json', timeout=30) as response:
        metadata_bytes = response.read()
    metadata = json.loads(metadata_bytes)
    choices = [r for r in metadata['urls'] if r['filename'].endswith('win_amd64.whl') and r['packagetype'] == 'bdist_wheel']
    if len(choices) != 1: raise ValueError('Exact official Windows AMD64 wheel required')
    wheel_record = choices[0]
    if urlparse(wheel_record['url']).hostname != 'files.pythonhosted.org': raise ValueError('Unexpected official wheel host')
    folder = S.OUT / 'assets/profiling/py_spy' / metadata['info']['version']
    folder.mkdir(parents=True, exist_ok=True)
    metadata_path = folder / 'official_pypi_metadata.json'
    if metadata_path.exists() and metadata_path.read_bytes() != metadata_bytes:
        metadata_path = folder / ('official_pypi_metadata_' + str(time.time_ns()) + '.json')
    metadata_path.write_bytes(metadata_bytes)
    wheel = folder / Path(wheel_record['filename']).name
    if not wheel.exists():
        with urllib.request.urlopen(wheel_record['url'], timeout=60) as response:
            wheel.write_bytes(response.read())
    if S.digest(wheel) != wheel_record['digests']['sha256'] or wheel.stat().st_size != wheel_record['size']:
        raise ValueError('Official wheel SHA256/size mismatch')
    with zipfile.ZipFile(wheel) as archive:
        members = [name for name in archive.namelist() if name.endswith('/py-spy.exe') or name == 'py-spy.exe']
        if len(members) != 1: raise ValueError('Exactly one profiler executable required')
        executable = folder / 'py-spy.exe'
        content = archive.read(members[0])
        if executable.exists() and executable.read_bytes() != content: raise ValueError('Staged binary changed')
        executable.write_bytes(content)
    path = S.OUT / 'supervised_engineering/CSBrain-lastblock-finetune' / ('readonly_profiler_setup_' + str(time.time_ns()) + '.json')
    S.save(path, {'kind': 'profiling_tool_setup', 'source': S.desc(Path(__file__)), 'status': 'complete',
                  'version': metadata['info']['version'], 'official_metadata': S.desc(metadata_path),
                  'official_wheel': S.desc(wheel), 'executable': S.desc(executable),
                  'seconds': time.perf_counter() - begin, 'model_constructions': 0, 'GPU_forward_calls': 0,
                  'backward_calls': 0, 'optimizer_updates': 0, 'formal_fits': 0,
                  'training_environment_packages_modified': False}, immutable=True)
    print(json.dumps({'setup': S.desc(path), 'executable': S.desc(executable)}))


if __name__ == '__main__': run()
