"""Workspace-local ASR caches and ffmpeg; no system PATH changes."""
import os
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]

def configure():
    for name,folder in {'HF_HOME':'hf','TRANSFORMERS_CACHE':'transformers','TORCH_HOME':'torch','XDG_CACHE_HOME':'xdg','MPLCONFIGDIR':'matplotlib','NLTK_DATA':'nltk'}.items():
        path=ROOT/'cache'/folder; path.mkdir(parents=True,exist_ok=True); os.environ[name]=str(path)
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    import imageio_ffmpeg
    executable=ROOT/'tools/ffmpeg.exe'; executable.parent.mkdir(exist_ok=True)
    if not executable.exists(): shutil.copyfile(imageio_ffmpeg.get_ffmpeg_exe(),executable)
    os.environ['PATH']=str(executable.parent)+os.pathsep+os.environ.get('PATH','')

def load():
    configure()
    import whisperx
    return whisperx.load_model(str(ROOT/'models/whisper_medium'),device='cpu',compute_type='float32',language='zh',local_files_only=True,download_root=str(ROOT/'cache/asr'),threads=4)
