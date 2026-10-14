"""Create a new workspace Linux environment using read-only old Torch packages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import venv

WORK = Path(__file__).resolve().parents[2]
ASSETS = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/assets'
ENV = ASSETS / 'eegmamba_linux_env'
BASE_PACKAGES = Path('/home/tridep/.venvs/tri-dep/lib/python3.12/site-packages')
WHEELS = {
    'mamba_ssm': {'release': 'https://api.github.com/repos/state-spaces/mamba/releases/tags/v2.2.4', 'filename': 'mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl'},
    'causal_conv1d': {'release': 'https://api.github.com/repos/Dao-AILab/causal-conv1d/releases/tags/v1.5.0.post8', 'filename': 'causal_conv1d-1.5.0.post8+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl'},
}


def open_url(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'EEG-AllCandidates-official-Mamba-wheel/1.0'}), timeout=60)


def main():
    if sys.platform != 'linux' or not ENV.resolve().is_relative_to(WORK.resolve()) or not BASE_PACKAGES.is_dir():
        raise RuntimeError('Expected workspace-local Linux environment with accepted read-only Torch base')
    if not (ENV / 'bin/python').exists():
        venv.EnvBuilder(with_pip=True, symlinks=True).create(ENV)
    site = ENV / 'lib/python3.12/site-packages'
    (site / 'accepted_tridep_readonly_base.pth').write_text(str(BASE_PACKAGES) + '\n', encoding='utf-8')
    manifest = {'environment': str(ENV.relative_to(WORK)), 'base_packages_read_only': str(BASE_PACKAGES), 'base_environment_modified': False, 'official_wheels': {}}
    for name, details in WHEELS.items():
        with open_url(details['release']) as response:
            release = json.load(response)
        asset = next(a for a in release['assets'] if a['name'] == details['filename'])
        target = ASSETS / 'eegmamba_linux_wheels' / details['filename']
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size != asset['size']:
            temporary = target.with_suffix(target.suffix + '.part')
            print('DOWNLOADING ' + name + ' ' + str(asset['size']), flush=True)
            with open_url(asset['browser_download_url']) as response, temporary.open('wb') as stream:
                while data := response.read(4 * 1024 * 1024):
                    stream.write(data)
            if temporary.stat().st_size != asset['size']:
                raise RuntimeError('Official release asset length mismatch')
            temporary.replace(target)
        with target.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest['official_wheels'][name] = {'path': str(target.relative_to(WORK)), 'url': asset['browser_download_url'],
                                             'bytes': asset['size'], 'sha256': sha, 'release_id': release['id'], 'asset_id': asset['id']}
        print('INSTALLING ' + name, flush=True)
        subprocess.run([str(ENV / 'bin/python'), '-m', 'pip', 'install', '--no-deps', str(target)], check=True)
        (ASSETS / 'eegmamba_linux_runtime.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('READY ' + str(ENV / 'bin/python'), flush=True)


if __name__ == '__main__':
    main()
