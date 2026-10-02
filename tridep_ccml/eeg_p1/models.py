from __future__ import annotations
from .common import *
from torch import nn
from tridep_ccml.eeg_replacement.models import R1Model, REGIONS
from tridep_ccml.eeg_replacement.common import REVISION


class SE(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(channels,max(1,channels//2)),nn.LeakyReLU(.1),
                                 nn.Linear(max(1,channels//2),channels),nn.Sigmoid())

    def forward(self,x):
        return x*self.net(x.mean(-1))[:,:,None]


class SparNetP1(nn.Module):
    def __init__(self):
        super().__init__()
        names = [r['target_name'] for r in read_json(REVISION)['selection']]
        self.region_indices = [[names.index(n) for n in group] for group in REGIONS.values()]
        self.regions = nn.ModuleList([nn.Sequential(SE(len(idx)),nn.Conv1d(len(idx),4,5),
            nn.LeakyReLU(.1),nn.MaxPool1d(2,2)) for idx in self.region_indices])
        self.global_conv = nn.Sequential(SE(20),nn.Conv1d(20,4,2),nn.LeakyReLU(.1),nn.MaxPool1d(2,2))
        self.interface = nn.Sequential(nn.Linear(188,128),nn.LeakyReLU(.1))
        for layer in self.modules():
            if isinstance(layer,(nn.Linear,nn.Conv1d)):
                nn.init.kaiming_uniform_(layer.weight,a=.1,nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    @staticmethod
    def spectrum(x):
        return 2*torch.fft.rfft(x,dim=-1,norm='ortho').abs()[...,5:201]

    def window_features(self,spectrum):
        branches = [net(spectrum[:,idx,:]) for net,idx in zip(self.regions,self.region_indices)]
        z = self.global_conv(torch.cat(branches,1)).flatten(1)
        require(z.shape[1]==188,'P1 terminal shape mismatch')
        return z

    def from_spectrum(self,spectrum):
        require(spectrum.shape[1:]==(60,19,196),'P1 spectral axes mismatch')
        z = self.window_features(spectrum.flatten(0,1)).reshape(len(spectrum),60,188).mean(1)
        return self.interface(z)

    def forward(self,x):
        require(x.shape[1:]==(60,19,1000),'P1 raw axes mismatch')
        return self.from_spectrum(self.spectrum(x))


class P1Model(R1Model):
    def __init__(self,condition,seed):
        super().__init__(condition,seed)
        if condition in ['U2','C2']:
            # Retain exactly the R1 classifier/evidence and non-EEG initial states.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+20000)
                self.eeg = SparNetP1()


def structure():
    return {'name':'SparNet-inspired-19ch-P1','augmentation':False,
        'spectrum':'2*abs(rfft(x,norm=ortho))[5:201]; fixed sqrt(1000) times R1 spectrum',
        'frequency_hz':[1,40],'frequency_points':196,'FFT_length':1000,
        'region_width':4,'global_width':4,'regional_kernel':5,'global_kernel':2,
        'activation':'LeakyReLU(0.1), including SE hidden layer and 128D interface',
        'initialization':'Kaiming uniform a=0.1 for encoder Conv/Linear; zero biases',
        'SE_ratio':2,'region_mapping':REGIONS,'window_output':[4,47],
        'aggregation':'188D per window, mean all 60 windows, Linear(188,128), LeakyReLU(.1)',
        'shared_heads':'exact original R1 U2/C2 heads; all C shared non-EEG initialization identical',
        'parameters':{c:P1Model(c,11).counts() for c in CONDITIONS},
        'scope':'Engineering repair bundle, not single-factor architecture attribution or original SparNet reproduction'}
