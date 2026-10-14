"""LUNA readout unchanged; preserve the author's fixed rotary frequencies."""
from __future__ import annotations

from pathlib import Path

from . import luna as V1
from ._author_support import WORK, digest, require, select_trainability

FEATURE_DIM = V1.FEATURE_DIM
INPUT_CONTRACT = V1.INPUT_CONTRACT


def exact_finetune_parameter_names(model):
    names = V1.exact_finetune_parameter_names(model)
    fixed = f'blocks.{len(model.blocks)-1}.attn.rotary_emb.freqs'
    require(fixed in names and model.blocks[-1].attn.rotary_emb.learned_freq is False,
            'Expected the formal author fixed rotary-frequency parameter')
    return [name for name in names if name != fixed]


def load_encoder(device='cuda', trainable=False):
    model, audit = V1.load_encoder(device=device, trainable=False)
    names = exact_finetune_parameter_names(model)
    select_trainability(model, trainable, names)
    fixed = f'blocks.{len(model.blocks)-1}.attn.rotary_emb.freqs'
    audit.update(adapter_version=2, finetune_parameter_names=names,
                 fixed_author_parameter_names=[fixed], learned_rotary_frequencies=False,
                 repair_reason='author learned_freq=False returns detached cached rotary frequencies; keep original fixed frequency rather than unfreezing it',
                 forward_and_readout_math_changed=False)
    audit['implementation_sources'] = [{'path': Path(path).resolve().relative_to(WORK).as_posix(), 'sha256': digest(path)}
                                       for path in (V1.__file__, __file__, Path(__file__).parent / '_author_support.py')]
    return model, audit


def adapt_raw(raw):
    return V1.adapt_raw(raw)


def encode_windows(model, tensor):
    return V1.encode_windows(model, tensor)
