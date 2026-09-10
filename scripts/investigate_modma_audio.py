"""Read-only corruption diagnostics. Does not invent headers or alter raw data."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import wave
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
OUT = ROOT / 'recovery'
RAW = WORK / 'dataset/modma/raw/modma-audio-group-6-dsai/MODMA_AUDIO Datatset/audio_lanzhou_2015/02010004'

def entropy(data):
    return -sum((n/len(data))*math.log2(n/len(data)) for n in Counter(data).values())

def inspect(path):
    data = path.read_bytes()
    row = {'path': str(path.relative_to(WORK)), 'bytes': len(data),
           'sha256': hashlib.sha256(data).hexdigest(), 'first_64_hex': data[:64].hex(),
           'entropy_bits_per_byte': entropy(data), 'zero_fraction': data.count(0)/len(data),
           'riff_offset': data.find(b'RIFF'), 'wave_offset': data.find(b'WAVE'),
           'blocks': [{'offset': start, 'bytes': len(data[start:start+65536]),
                       'entropy': entropy(data[start:start+65536]),
                       'zero_fraction': data[start:start+65536].count(0)/len(data[start:start+65536])}
                      for start in range(0, len(data), 65536)]}
    try:
        with wave.open(io.BytesIO(data), 'rb') as wav:
            n = wav.getnframes()
            pcm = wav.readframes(n)
            row['wave'] = {'valid': len(pcm)==n*wav.getsampwidth()*wav.getnchannels(),
                           'sample_rate': wav.getframerate(), 'sample_width': wav.getsampwidth(),
                           'channels': wav.getnchannels(), 'frames': n}
    except Exception as exc:
        row['wave'] = {'valid': False, 'error': str(exc)}
    ffmpeg = ROOT / 'tools/ffmpeg.exe'
    try:
        proc = subprocess.run([str(ffmpeg), '-v', 'error', '-nostdin', '-i', str(path),
                               '-f', 'null', '-'], capture_output=True, timeout=20,
                              text=True, encoding='utf-8', errors='replace')
        row['ffmpeg'] = {'returncode': proc.returncode, 'diagnostic': proc.stderr[-3000:]}
    except subprocess.TimeoutExpired:
        row['ffmpeg'] = {'timeout': True}
    return row

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(WORK/'dataset/modma/modma-audio-group-6-dsai.zip') as archive:
        for q in (23,24,25,26,27,28,29):
            name = f'{q:02d}.wav'
            row = inspect(RAW/name)
            member = 'MODMA_AUDIO Datatset/audio_lanzhou_2015/02010004/' + name
            info = archive.getinfo(member)
            original = archive.read(member)  # zipfile also checks member CRC
            row['archive'] = {'member': member, 'crc32': f'{info.CRC:08x}',
                              'crc_passed': zlib.crc32(original)==info.CRC,
                              'raw_exactly_matches_archive': original==(RAW/name).read_bytes(),
                              'compressed_bytes': info.compress_size, 'flags': info.flag_bits,
                              'encrypted': bool(info.flag_bits & 1), 'compression': info.compress_type}
            rows.append(row)
    with zipfile.ZipFile(WORK/'dataset/modma/modma-dataset-2015.zip') as archive:
        alternate = [i.filename for i in archive.infolist()
                     if 'audio_lanzhou_2015/02010004_' in i.filename]
    report = {'checked_utc': datetime.now(timezone.utc).isoformat(),
              'subject_id': '02010004', 'invalid_questions': ['24','25','26','27','28'],
              'controls': ['23','29'], 'files': rows,
              'alternate_eeglab_members_for_subject': alternate,
              'alternate_absent_questions': [str(q) for q in range(24,29)
                    if not any(f'02010004_{q}.' in s for s in alternate)],
              'raw_files_modified': False, 'recovered_full_recordings': 0,
              'interpretation': 'CRC equality rules out extraction mismatch; uniform byte entropy alone does not prove encryption or recover original samples. Lower-entropy blocks do not establish a complete WAV or its missing header.'}
    (OUT/'local_diagnostics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps([{'file':Path(r['path']).name, 'wave_valid':r['wave']['valid'],
                      'ffmpeg':r['ffmpeg'].get('returncode'),
                      'archive_equal':r['archive']['raw_exactly_matches_archive'],
                      'entropy_min':min(b['entropy'] for b in r['blocks']),
                      'entropy_max':max(b['entropy'] for b in r['blocks'])} for r in rows], indent=2))

if __name__ == '__main__':
    main()
