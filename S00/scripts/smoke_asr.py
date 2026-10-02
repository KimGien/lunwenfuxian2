"""Local-only ASR check on a ten-second excerpt; never a classification run."""
import hashlib
import json
import sys
import time
from asr_runtime import load,ROOT
from prepare import save,digest,WORK
sys.stdout.reconfigure(encoding='utf-8')

def main():
    start=time.perf_counter(); model=load()
    import whisperx
    source=ROOT/'staged/02010002/audio/01.wav'
    audio=whisperx.load_audio(str(source))[:16000*10]
    result=model.transcribe(audio,batch_size=1,language='zh',task='transcribe')
    assert result['language']=='zh' and isinstance(result['segments'],list)
    report={'passed':True,'model':'Systran/faster-whisper-medium','device':'cpu','compute_type':'float32','language':'zh','input_seconds':len(audio)/16000,'source_path':str(source.relative_to(WORK)),'source_sha256':digest(source),'segments':len(result['segments']),'transcript_sha256':hashlib.sha256(json.dumps(result,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),'elapsed_seconds':time.perf_counter()-start,'scope':'10-second loading/decoding test only; not corpus transcription or diagnosis'}
    save(ROOT/'reports/asr_smoke.json',report); print(json.dumps(report,indent=2))

if __name__=='__main__':
    try: main()
    except Exception as exc:
        save(ROOT/'reports/asr_smoke.json',{'passed':False,'error':str(exc)})
        raise
