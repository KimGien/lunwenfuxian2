"""Retrieve immutable author snapshots and weights; never import or train them.

All destinations are under the new all-candidates module and experiment tree.
Existing files are verified/reused. Partial downloads never replace a final file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.parse
import urllib.request
import zipfile

WORK = Path(__file__).resolve().parents[2]
VENDOR = Path(__file__).resolve().parent / 'vendor'
ASSETS = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/assets'
INDEX = ASSETS / 'download_manifest.json'
REPOS = {
    'csbrain': 'yuchen2199/CSBrain',
    'reve': 'elouayas/reve_eeg',
    'luna': 'pulp-bio/BioFoundation',
    'codebrain': 'jingyingma01/CodeBrain',
    'eegconformer': 'eeyhsong/EEG-Conformer',
    'biot': 'ycq091044/BIOT',
    'eeg2rep': 'Navidfoumani/EEG2Rep',
    'eegmamba': 'wjq-learning/EEGMamba',
}
HUBS = {'reve': 'brain-bzh/reve-base', 'reve_positions': 'brain-bzh/reve-positions', 'luna': 'PulpBio/LUNA',
        'codebrain': 'YjMajy/CodeBrain', 'eegmamba': 'weighting666/EEGMamba'}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        'User-Agent': 'EEG-AllCandidates-public-resource-audit/1.0'}), timeout=45)


def api(url):
    with request(url) as response:
        return json.load(response)


def fetch(url, path):
    path = path.resolve()
    if not path.is_relative_to(WORK):
        raise ValueError('Destination outside workspace')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + '.part')
        with request(url) as response, temporary.open('wb') as stream:
            size = 0
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > 2 * 1024 ** 3:
                    raise ValueError('Unexpected download over 2 GiB')
                stream.write(block)
        temporary.replace(path)
    return {'path': path.relative_to(WORK).as_posix(), 'url': url,
            'sha256': sha(path), 'bytes': path.stat().st_size}


def source(name, repo):
    binding_path = VENDOR / name / 'author_source_binding.json'
    if binding_path.exists():
        existing = json.loads(binding_path.read_text(encoding='utf-8'))
        for f in existing['files']:
            if sha(WORK / f['path']) != f['sha256']:
                raise ValueError('Existing author source changed: ' + f['path'])
        return existing
    info = api('https://api.github.com/repos/' + repo)
    revision = api('https://api.github.com/repos/' + repo + '/commits/' + info['default_branch'])['sha']
    archive = fetch('https://codeload.github.com/' + repo + '/zip/' + revision,
                    ASSETS / 'source_archives' / (name + '-' + revision + '.zip'))
    destination = VENDOR / name
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    with zipfile.ZipFile(WORK / archive['path']) as z:
        for member in z.infolist():
            if member.is_dir():
                continue
            parts = Path(member.filename).parts
            if len(parts) < 2 or '..' in parts:
                raise ValueError('Unsafe author archive member')
            path = (destination / Path(*parts[1:])).resolve()
            if not path.is_relative_to(destination.resolve()):
                raise ValueError('Unsafe source destination')
            path.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(member)
            if path.exists() and path.read_bytes() != data:
                raise ValueError('Refusing to overwrite different source: ' + str(path))
            path.write_bytes(data)
            files.append({'path': path.relative_to(WORK).as_posix(),
                          'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
    result = {'repository': 'https://github.com/' + repo, 'revision': revision,
              'source_modified': False, 'archive': archive, 'files': files}
    save(binding_path, result)
    return result


def hub_inventory(name, repo):
    info = api('https://huggingface.co/api/models/' + repo)
    result = {'repository': repo, 'revision': info['sha'],
              'siblings': [f['rfilename'] for f in info['siblings']]}
    save(ASSETS / name / 'hub_inventory.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('sources', 'hub-inventory', 'weights', 'csbrain-weight', 'biot-weight'))
    parser.add_argument('--names', nargs='*')
    args = parser.parse_args()
    state = json.loads(INDEX.read_text(encoding='utf-8')) if INDEX.exists() else {'sources': {}, 'hub_inventory': {}, 'weights': {}, 'failures': []}
    if args.phase == 'csbrain-weight':
        file_id = '1Z67je1HQhClyG9XER_zJzFMAKSVNmnF0'
        item = fetch('https://drive.usercontent.google.com/download?id=' + file_id + '&export=download&authuser=0&confirm=t', ASSETS / 'csbrain/CSBrain.pth')
        if item['bytes'] != 35722626 or not zipfile.is_zipfile(WORK / item['path']):
            raise ValueError('CSBrain public file does not match directory bytes / Torch archive')
        state['weights']['csbrain'] = {'folder': 'https://drive.google.com/drive/folders/1je-1TtdHv6klcd-kTlPNkiA1wLrxybva', 'public_file_id': file_id, 'files': [item]}
        save(INDEX, state)
        print('DONE weights csbrain', flush=True)
        return
    if args.phase == 'biot-weight':
        source_path = VENDOR / 'biot/pretrained-models/EEG-PREST-16-channels.ckpt'
        target = ASSETS / 'biot/EEG-PREST-16-channels.ckpt'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
        item = {'path': target.relative_to(WORK).as_posix(), 'sha256': sha(target), 'bytes': target.stat().st_size,
                'url': 'https://raw.githubusercontent.com/' + REPOS['biot'] + '/' + state['sources']['biot']['revision'] + '/pretrained-models/EEG-PREST-16-channels.ckpt',
                'from_verified_source_archive': source_path.relative_to(WORK).as_posix()}
        state['weights']['biot'] = {'revision': state['sources']['biot']['revision'], 'files': [item]}
        save(INDEX, state)
        print('DONE weights biot', flush=True)
        return
    mapping = REPOS if args.phase == 'sources' else HUBS
    names = args.names or list(mapping)
    for name in names:
        print('START ' + args.phase + ' ' + name, flush=True)
        try:
            if args.phase == 'sources':
                state['sources'][name] = source(name, mapping[name])
            elif args.phase == 'hub-inventory':
                state['hub_inventory'][name] = hub_inventory(name, mapping[name])
            else:
                inventory = state['hub_inventory'].get(name) or hub_inventory(name, mapping[name])
                selected = []
                for f in inventory['siblings']:
                    lower = f.lower()
                    if name == 'reve':
                        keep = f.endswith(('.json', '.py', '.safetensors')) or lower in ('readme.md', 'license')
                    elif name == 'luna':
                        keep = ('base' in lower and f.endswith('.safetensors')) or f.endswith(('.py', '.json', '.yaml')) or lower in ('readme.md', 'license')
                    elif name == 'codebrain':
                        keep = f.split('/')[-1].lower() == 'codebrain.pth' or f.endswith(('.json', '.py')) or lower == 'readme.md'
                    else:
                        keep = f.endswith(('.pth', '.pt', '.json', '.py', '.safetensors')) or lower == 'readme.md'
                    if keep:
                        selected.append(fetch('https://huggingface.co/' + mapping[name] + '/resolve/' + inventory['revision'] + '/' + urllib.parse.quote(f), ASSETS / name / f))
                state['weights'][name] = {'repository': mapping[name], 'revision': inventory['revision'], 'files': selected}
            print('DONE ' + args.phase + ' ' + name, flush=True)
        except Exception as e:
            failure = {'phase': args.phase, 'name': name, 'error': str(e), 'time': time.time()}
            state['failures'].append(failure)
            print('FAILED ' + json.dumps(failure), flush=True)
        save(INDEX, state)


if __name__ == '__main__':
    main()
