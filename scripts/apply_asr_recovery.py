"""Apply the smallest successful amplitude retry, retaining original ASR."""
import json
import shutil
from prepare import ROOT,save,digest

def main():
    evidence=ROOT/'reports/asr_empty_retry_02010036_22.json'
    report=json.loads(evidence.read_text(encoding='utf-8'))
    target=ROOT/'transcripts/B0-37/first_segment_code/02010036/22.json'
    raw=ROOT/'staged/02010036/audio/22.wav'
    old=json.loads(target.read_text(encoding='utf-8'))
    if not old['empty']:raise ValueError('Refusing to overwrite a nonempty transcript')
    if digest(raw)!=report['source_sha256'] or old['source_sha256']!=report['source_sha256']:
        raise ValueError('Recovery source changed')
    candidates=sorted(report['candidates'],key=lambda r:r['gain'])
    selected=next((c for c in candidates if c['segments'] and c['segments'][0]['text'].strip()),None)
    if selected is None:raise ValueError('No successful amplitude retry')
    archive=ROOT/'archive/asr_original_empty/02010036/22.json'
    archive.parent.mkdir(parents=True,exist_ok=True)
    if archive.exists() and digest(archive)!=digest(target):raise ValueError('Original archive differs')
    shutil.copy2(target,archive)
    original_hash=digest(target)
    old.update(segments=selected['segments'],answer_text=selected['segments'][0]['text'],
        all_segments_text=''.join(s['text'] for s in selected['segments']),empty=False,
        recovery={'method':'amplitude_only_retry_smallest_successful_gain','gain':selected['gain'],
            'selection_rule':'Test gains1,10,100 in ascending order; choose first nonempty result, no label/score selection.',
            'evidence_sha256':digest(evidence),'original_transcript_sha256':original_hash,
            'raw_waveform_unchanged':True,'human_verified':False,
            'limitation':'ASR spellings differ slightly between gain10 and gain100. Store model output verbatim; no manual reconstruction.'})
    temporary=target.with_suffix('.json.part');save(temporary,old);temporary.replace(target)
    print('Recovered one previously empty recording using gain='+str(selected['gain'])+'; original audio/transcript retained.')

if __name__=='__main__':main()
