import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .cbramod import CBraMod

class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )

        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}') #or cpu, depends
            ckpt = torch.load(param.foundation_dir, map_location=torch.device('cpu'))

            # Remove prefix like "backbone." if present
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            ckpt = {k.replace("backbone.", ""): v for k, v in ckpt.items()}
            self.backbone.load_state_dict(ckpt, strict=False)

        # Disable projection head
        self.backbone.proj_out = nn.Identity()

        self.classifier = nn.Identity()  

    def forward(self, x):
        return self.backbone(x)

class Params:
    use_pretrained_weights = True
    cuda = 0
    foundation_dir = 'pretrained_weights/pretrained_weights.pth'
    classifier = ''
    dropout = 0.1
