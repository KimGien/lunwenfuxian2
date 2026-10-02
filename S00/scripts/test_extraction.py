"""Regression checks against the vendor's real segmentation/encoder behavior."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from extract_b0 import code_windows,speech_preprocessing
segment_audio=speech_preprocessing().segment_audio
from lib.models.encoders.cnn_bigru import BiGRUAttentionEncoder
from smoke_models import seed

class ExtractionTests(unittest.TestCase):
    def test_windows_match_written_upstream_pcm(self):
        for seconds in (3,5,10,11.3):
            with self.subTest(seconds=seconds),tempfile.TemporaryDirectory() as tmp:
                y=np.random.default_rng(19).uniform(-.9,.9,int(seconds*16000)).astype(np.float32)
                segment_audio(y,16000,'q',Path(tmp))
                windows=code_windows(len(y))
                self.assertEqual(len(list(Path(tmp).glob('q_part*.wav'))),len(windows))
                for i,(start,end) in enumerate(windows):
                    expected=Path(tmp)/'expected.wav';sf.write(expected,y[start:end],16000)
                    self.assertEqual(expected.read_bytes(),(Path(tmp)/f'q_part{i+1}.wav').read_bytes())
    def test_short_and_duplicate_are_disclosed(self):
        self.assertEqual(code_windows(48000),[])
        windows=code_windows(160000)
        self.assertEqual(windows,[(0,80000),(40000,120000),(80000,160000),(80000,160000)])
    def test_seed_reconstructs_random_weights_and_singleton_attention(self):
        torch.set_num_threads(2)
        seed(11);a=BiGRUAttentionEncoder(input_dim=768).eval()
        seed(11);b=BiGRUAttentionEncoder(input_dim=768).eval()
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        inputs=torch.arange(4*768,dtype=torch.float32).reshape(4,768)/1000
        seen=[]
        handle=a.attention.register_forward_hook(lambda m,i,o:seen.append(torch.softmax(o,dim=1)))
        with torch.no_grad():
            actual=a(inputs);independent=torch.cat([a(x.unsqueeze(0)) for x in inputs])
        handle.remove()
        self.assertEqual(tuple(actual.shape),(4,512))
        self.assertTrue(all(torch.equal(x,torch.ones_like(x)) for x in seen))
        torch.testing.assert_close(actual,independent)

if __name__=='__main__': unittest.main(verbosity=2)
