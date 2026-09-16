"""Run the authorized local B0 stages, waiting for an already-running ASR process.

Writes progress and logs under reports. Never substitutes failed/empty transcripts.
No scheduler, network requests, or remote publication is involved.
"""
import argparse
import ctypes
from datetime import datetime,timezone
import json
import os
import subprocess
import sys
import time
from prepare import ROOT,save

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--asr-pid',type=int,required=True)
    parser.add_argument('--train',action='store_true')
    args=parser.parse_args()
    status={'pid':os.getpid(),'asr_pid':args.asr_pid,'started_utc':datetime.now(timezone.utc).isoformat(),
        'completed_stages':[],'training_requested':args.train}
    progress=ROOT/'reports/b0_execution_progress.json'
    def update(stage,state='running',**extra):
        status.update(stage=stage,status=state,updated_utc=datetime.now(timezone.utc).isoformat(),**extra)
        save(progress,status);print(json.dumps(status),flush=True)
    def run(stage,script,*arguments):
        update(stage)
        log=ROOT/'reports'/f'b0_{stage}.log'
        with log.open('a',encoding='utf-8') as output:
            result=subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script),*arguments],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'{stage} failed (exit {result.returncode}); inspect {log.name}')
        status['completed_stages'].append(stage)
    try:
        run('eeg_features','extract_b0.py','--modality','eeg')
        run('speech_features','extract_b0.py','--modality','speech')
        update('waiting_for_asr')
        # Keep a process handle, rather than trusting a potentially reused process ID.
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
        kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        handle=kernel.OpenProcess(0x00100000,False,args.asr_pid)
        try:
            while True:
                p=ROOT/'reports/transcription_progress.json'
                try:
                    asr=json.loads(p.read_text()) if p.exists() else {}
                except json.JSONDecodeError:
                    # The running ASR writer may be between truncate and write.
                    time.sleep(.2)
                    continue
                if asr.get('status')=='complete' and asr.get('completed')==1073: break
                if asr.get('status')=='needs_empty_review': raise RuntimeError('ASR empty results require review; text features and training stopped')
                if not handle or kernel.WaitForSingleObject(handle,0)!=258:
                    raise RuntimeError('ASR process ended before verified completion; resume transcription first')
                time.sleep(5)
        finally:
            if handle: kernel.CloseHandle(handle)
        run('text_features','extract_b0.py','--modality','text')
        for value in (11,22,33): run(f'preflight_{value}','run_b0.py','--seed',str(value))
        if args.train:
            for value in (11,22,33): run(f'train_{value}','run_b0.py','--seed',str(value),'--train')
        update('training_complete' if args.train else 'features_ready','complete')
        subprocess.run([sys.executable,str(ROOT/'scripts/finalize.py')],cwd=ROOT,check=True)
    except Exception as exc:
        update(status.get('stage','startup'),'failed',error=str(exc));raise

if __name__=='__main__': main()
