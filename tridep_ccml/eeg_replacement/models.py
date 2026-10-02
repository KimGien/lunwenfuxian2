"""Fixed architectures, paired non-EEG initialization, subject-level outputs."""
from __future__ import annotations
import math
import sys
from .common import *
from torch import nn
from torch.nn import functional as F
from tridep_ccml.ccml import ThreeViewHeads
from tridep_ccml.train import make_initial_state

sys.path.insert(0, str(WORK/'S00/upstream-dev'))
from lib.models.models import EEGConvPoolClassifier

# Array positions come exclusively from the verified source-label revision.
# FCz/CPz are explicitly assigned to the central transition group, not renamed Cz/Fz.
REGIONS = {'frontal': ['FP2', 'FP1', 'F7', 'F3', 'F4', 'F8'],
           'central': ['FCz', 'C3', 'C4', 'CPz'],
           'temporal': ['T3', 'T4', 'T5', 'T6'],
           'parietal': ['P3', 'Pz', 'P4'], 'occipital': ['O1', 'O2']}


@torch.no_grad()
def max_norm_(weight, limit):
    axes = tuple(range(1, weight.ndim))
    norm = torch.linalg.vector_norm(weight, dim=axes, keepdim=True)
    weight.mul_((limit / norm.clamp_min(1e-12)).clamp(max=1.))


class EEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = nn.Conv2d(1, 4, (1, 100), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(4, eps=1e-3, momentum=.01)
        self.spatial = nn.Conv2d(4, 8, (19, 1), groups=4, bias=False)
        self.bn2 = nn.BatchNorm2d(8, eps=1e-3, momentum=.01)
        self.drop1 = nn.Dropout(.25)
        self.depthwise = nn.Conv2d(8, 8, (1, 25), groups=8, padding='same', bias=False)
        self.pointwise = nn.Conv2d(8, 8, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(8, eps=1e-3, momentum=.01)
        self.drop2 = nn.Dropout(.25)
        self.interface = nn.Sequential(nn.Linear(248, 128), nn.ReLU())

    def window_features(self, x):
        x = self.bn1(self.temporal(x[:, None]))
        x = self.drop1(F.avg_pool2d(F.elu(self.bn2(self.spatial(x))), (1, 4)))
        x = self.drop2(F.avg_pool2d(F.elu(self.bn3(self.pointwise(self.depthwise(x)))), (1, 8)))
        require(x.shape[1:] == (8, 1, 31), 'EEGNet terminal axes mismatch')
        return x.flatten(1)

    def forward(self, x):
        require(x.ndim == 4 and x.shape[1:] == (60, 19, 1000), 'EEGNet expects 60x19x1000 per subject')
        # Full subject batch through BN; no microbatching or dropped windows.
        z = self.window_features(x.flatten(0, 1)).reshape(len(x), 60, 248).mean(1)
        return self.interface(z)


class SE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(channels, max(1, channels//2)), nn.ReLU(),
                                 nn.Linear(max(1, channels//2), channels), nn.Sigmoid())

    def forward(self, x):
        return x * self.net(x.mean(-1))[:, :, None]


class SparNet(nn.Module):
    def __init__(self):
        super().__init__()
        names = [r['target_name'] for r in read_json(REVISION)['selection']]
        self.region_indices = [[names.index(n) for n in group] for group in REGIONS.values()]
        self.regions = nn.ModuleList([nn.Sequential(SE(len(idx)), nn.Conv1d(len(idx), 1, 5),
                                                   nn.ReLU(), nn.MaxPool1d(2, 2))
                                      for idx in self.region_indices])
        self.global_conv = nn.Sequential(SE(5), nn.Conv1d(5, 1, 2), nn.ReLU(), nn.MaxPool1d(2, 2))
        self.interface = nn.Sequential(nn.Linear(47, 128), nn.ReLU())

    @staticmethod
    def spectrum(x):
        # N=1000, fs=200 => bins k=5..200 are exactly 1..40 Hz, 196 bins.
        return 2 * torch.fft.rfft(x, dim=-1, norm='forward').abs()[..., 5:201]

    def forward(self, x):
        require(x.shape[1:] == (60, 19, 1000), 'SparNet raw axes mismatch')
        spectrum = self.spectrum(x.flatten(0, 1))
        branches = [net(spectrum[:, idx, :]) for net, idx in zip(self.regions, self.region_indices)]
        z = self.global_conv(torch.cat(branches, 1)).flatten(1)
        require(z.shape[1] == 47, 'SparNet frequency axis mismatch')
        return self.interface(z.reshape(len(x), 60, 47).mean(1))


class TokenPool(nn.Module):
    def __init__(self, seed, attention):
        super().__init__()
        # Draw shared projection and interface before attention-only parameters.
        self.projection = nn.Linear(200, 32)
        self.interface = nn.Sequential(nn.Linear(32, 128), nn.ReLU())
        self.attention = attention
        if attention:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 44000)
                self.query = nn.Parameter(torch.randn(2, 16) * .02)

    def pool(self, tokens):
        mean = tokens.mean(-2)
        if not self.attention:
            return mean
        # 95 positions (19 channels x 5 seconds), 2 disjoint 16-D heads.
        heads = tokens.reshape(*tokens.shape[:-1], 2, 16)
        weights = ((heads * self.query).sum(-1)/math.sqrt(16)).softmax(-2)
        attended = (heads * weights[..., None]).sum(-3).flatten(-2)
        return .5*mean + .5*attended

    def forward(self, x):
        require(x.shape[1:] == (60, 19, 5, 200), 'Token axes mismatch')
        z = self.projection(x.flatten(2, 3))
        return self.interface(self.pool(z).mean(1))


class R1Model(nn.Module):
    def __init__(self, condition, seed):
        super().__init__()
        require(condition in CONDITIONS, 'Unknown condition')
        self.condition = condition
        self.joint = condition.startswith('C')
        # Exactly reproduce S-control's RNG order for EEG/speech/text projections.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            base = nn.ModuleDict({m: nn.Sequential(nn.Linear(d, 128), nn.ReLU())
                                 for m, d in zip(MODALITIES, [200, 768, 768])})
        if self.joint:
            self.speech = base['speech']
            self.text = base['text']
            self.normalizer = nn.LayerNorm(128, eps=1e-5, elementwise_affine=False)
            self.heads = ThreeViewHeads()
            self.heads.load_state_dict(make_initial_state(seed), strict=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + {'0':0, '1':10000, '2':20000, '3':30000, '4':30000}[condition[1]])
            if condition == 'U0':
                self.eeg = EEGConvPoolClassifier(input_dim=200, seq_len=60, dropout=.4)
            elif condition == 'C0':
                self.eeg = base['eeg']
            elif condition in ['U1', 'C1']:
                self.eeg = EEGNet()
            elif condition in ['U2', 'C2']:
                self.eeg = SparNet()
            else:
                self.eeg = TokenPool(seed, attention=condition == 'C4')
            if not self.joint and condition != 'U0':
                self.classifier = nn.Linear(128, 2)

    def representations(self, inputs):
        if self.condition == 'U0':
            x = self.eeg.conv(inputs['eeg'].transpose(1, 2))
            return self.eeg.classifier[:-1](x)
        eeg = self.eeg(inputs['eeg'])
        if not self.joint:
            return eeg
        return torch.stack([eeg, self.speech(inputs['speech']), self.text(inputs['text'])], 1)

    def forward(self, inputs):
        z = self.representations(inputs)
        if self.joint:
            return F.softplus(self.heads(self.normalizer(z)))
        return (self.eeg.classifier[-1] if self.condition == 'U0' else self.classifier)(z)

    def optimizer(self):
        if not self.joint:
            return torch.optim.Adamax(self.parameters(), lr=1e-3, weight_decay=1e-4)
        head_ids = {id(p) for p in self.heads.parameters()}
        return torch.optim.Adamax([
            {'params': [p for p in self.parameters() if id(p) not in head_ids], 'lr':1e-4},
            {'params': self.heads.parameters(), 'lr':1e-3}], weight_decay=0.)

    def constrain(self):
        if self.condition in ['U1', 'C1']:
            max_norm_(self.eeg.spatial.weight, 1.)
        if self.condition == 'U1':
            max_norm_(self.classifier.weight, .25)

    def shared_state(self):
        return {k: v for k, v in self.state_dict().items() if k.startswith(('speech.', 'text.', 'heads.'))}

    def counts(self):
        eeg = sum(p.numel() for p in self.eeg.parameters())
        if hasattr(self, 'classifier'):
            eeg += sum(p.numel() for p in self.classifier.parameters())
        if self.joint:
            eeg += sum(p.numel() for p in self.heads.heads['eeg'].parameters())
        return {'total': sum(p.numel() for p in self.parameters()), 'eeg_including_head': eeg,
                'all_trainable': all(p.requires_grad for p in self.parameters()), 'frozen_backbones': 'offline label-free caches'}


def structure_manifest():
    selection = read_json(REVISION)['selection']
    by_name = {r['target_name']: r for r in selection}
    return {'conditions': CONDITIONS, 'parameters': {c: R1Model(c, 11).counts() for c in CONDITIONS},
        'augmentation': False, 'EEGNet': {'F1':4, 'D':2, 'F2':8, 'temporal_kernel':100,
            'separable_kernel':25, 'same_padding':True, 'pool':[4,8], 'dropout':.25,
            'batchnorm_eps':1e-3, 'batchnorm_momentum':.01, 'window_output':[8,1,31],
            'aggregation':'60-window mean before Linear(248,128)+ReLU', 'spatial_max_norm':1.,
            'U1_classifier_max_norm':.25, 'C1_evidence_head_max_norm':None},
        'SparNet': {'name':'SparNet-inspired-19ch', 'paper': descriptor(WORK/'baseline备选/21_SparNet_EEG_Depression_2022.pdf'),
            'paper_url':'https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2022.914823/full',
            'layers_source':'Table 1, use pooling stride 2 in layer table (prose stride 1 conflicts)',
            'spectrum':'2*abs(rfft(standardized 1000 samples, norm=forward))[5:201]; rectangular window; no log/power/extra denoising',
            'frequency_hz':[1.,40.], 'resolution_hz':.2, 'frequency_bins':196,
            'regional_conv':'SE(C,max(1,C//2),C); Conv1d(C,1,5) valid; ReLU; MaxPool(2,2)',
            'global_conv':'concat 5 regions; SE(5,2,5); Conv1d(5,1,2) valid; ReLU; MaxPool(2,2)',
            'shapes':[196,192,96,95,47], 'aggregation':'60-window mean then Linear(47,128)+ReLU',
            'SE_reduction':2, 'regions':{k:[by_name[n] for n in ns] for k,ns in REGIONS.items()},
            'mapping_sources':[descriptor(REVISION), descriptor(TC/'reports/ccml_v2_audit/CCML-v2-Audit输入核查报告.md')],
            'region_limitations':'Manufacturer approximate 10-10 labels, not subject-specific anatomy. FCz/CPz are transition electrodes assigned explicitly to central group. F8 follows verified E122 (not assumed E123). No GSN129 coordinates, interpolation or original 73-channel map used.',
            'adaptations':'19 channels; 5-second windows; 196 bins at native 0.2 Hz resolution; one regional filter each as 5-feature-map schematic; 128D subject interface; all fixed before outer score release'},
        'token_pool': {'name':'CBraMod+MHAP / STAMP-inspired', 'projection':[200,32],
            'positions_per_window':95, 'axis':'19 channels x 5 one-second tokens, never subjects/windows',
            'heads':2, 'head_dim':16, 'attention':'softmax(dot(projected_head,learned_query)/sqrt(16)) over 95 positions',
            'C3':'mean(projected_tokens,95)', 'C4':'0.5*mean(projected_tokens,95)+0.5*concat(two attention-pooled 16D heads)',
            'subject_aggregation':'mean over 60 windows then Linear(32,128)+ReLU',
            'shared_initialization':'projection/interface exactly shared C3/C4; all C speech/text projections and heads exactly shared',
            'token_replay_tolerance':{'rtol':1e-4,'atol':1e-4}, 'complete_STAMP_reproduction':False}}
