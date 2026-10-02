"""Diagnose empty ASR with explicit amplitude-only retry candidates; no promotion."""
import json
import numpy as np
from asr_runtime import ROOT,load
from prepare import save,digest

def main():
    model=load()
    import whisperx
    source=ROOT/'staged/02010036/audio/22.wav'
    audio=whisperx.load_audio(str(source))
    report={'subject_id':'02010036','question_id':'22','source_sha256':digest(source),
        'source_unchanged':True,'model_lock_sha256':digest(ROOT/'manifests/backbones.json'),
        'sample_rate':16000,'samples':len(audio),'peak':float(np.max(np.abs(audio))),
        'runtime':'whisperx_cpu_float32_batch1_zh_medium','candidates':[]}
    for gain in (1,10,100):
        candidate=audio*gain
        if np.max(np.abs(candidate))>1:raise ValueError('Retry gain would clip')
        result=model.transcribe(candidate,batch_size=1,language='zh',task='transcribe')
        report['candidates'].append({'gain':gain,'segments':result['segments']})
        save(ROOT/'reports/asr_empty_retry_02010036_22.json',report)
        print(json.dumps({'gain':gain,'segment_count':len(result['segments'])}),flush=True)

if __name__=='__main__':main()
