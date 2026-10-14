"""Author-bound mechanisms, adapted only to fixed 128-dimensional readouts."""
from __future__ import annotations
import importlib.util
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tridep_ccml import ccml
from .common import CODE

def load_reference(name, file):
    spec = importlib.util.spec_from_file_location('fusion_reference_' + name + '_' + file.replace('.', '_'),
                                                  CODE / 'vendor' / name / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Fusion(nn.Module):
    def __init__(self, method, views, n_train=27):
        super().__init__()
        self.method, self.views = method, views
        if method == 'CCML':
            self.heads = ccml.ThreeViewHeads(modalities=tuple('EST'[-views:]))
        elif method == 'MMDynamics':
            self.network = load_reference(method, 'model.py').MMDynamic([128]*views, [32], 2, .2)
        elif method in ('QMF', 'ECML'):
            self.heads = nn.ModuleList([nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(.2),
                                                      nn.Linear(32, 2)) for _ in range(views)])
        elif method == 'LMF':
            self.factors = nn.ParameterList([nn.Parameter(torch.empty(2, 129, 16)) for _ in range(views)])
            self.rank_weights = nn.Parameter(torch.empty(1, 2))
            self.bias = nn.Parameter(torch.zeros(1, 16))
            for p in [*self.factors, self.rank_weights]:
                nn.init.xavier_normal_(p)
            self.classifier = nn.Sequential(nn.ReLU(), nn.Dropout(.2), nn.Linear(16, 2))
        else:
            raise ValueError(method)
        # Author's cumulative CE history is float64 and train-only. It is durable.
        self.history = np.zeros((views, n_train), dtype=np.float64) if method == 'QMF' else None
        self.confidence_history = np.zeros((views, n_train), dtype=np.float64) if method == 'QMF' else None

    def forward(self, x):
        if self.method == 'CCML':
            evidence = self.heads.evidence(x)
            return {'p': ccml.evidence_dc(evidence, 1.25)['p'], 'evidence': evidence}
        if self.method == 'MMDynamics':
            logits = self.network.infer([x[:, i] for i in range(self.views)])
        elif self.method == 'LMF':
            interaction = None
            for i, factor in enumerate(self.factors):
                augmented = torch.cat([torch.ones_like(x[:, i, :1]), x[:, i]], dim=1)
                projected = torch.matmul(augmented, factor)  # rank x batch x output
                interaction = projected if interaction is None else interaction * projected
            fused = torch.matmul(self.rank_weights, interaction.permute(1, 0, 2)).reshape(-1, 16) + self.bias
            logits = self.classifier(fused)
        else:
            logits_v = torch.stack([self.heads[i](x[:, i]) for i in range(self.views)], dim=1)
            if self.method == 'QMF':
                # RGBD author path: confidence differentiates through dynamic fusion.
                quality = .1 * torch.logsumexp(logits_v, dim=-1)
                logits = (logits_v * quality.unsqueeze(-1)).sum(1)
                return {'p': logits.softmax(-1), 'logits': logits, 'logits_v': logits_v, 'quality': quality}
            evidence = F.softplus(logits_v)
            # Pinned RCML implements sequential average belief/evidence fusion,
            # with conflict reliability enforced by get_dc_loss, not DS fusion.
            final = evidence[:, 0]
            for i in range(1, self.views):
                final = (final + evidence[:, i]) / 2
            alpha = final + 1
            return {'p': alpha / alpha.sum(-1, keepdim=True), 'evidence': evidence, 'final': final}
        return {'p': logits.softmax(-1), 'logits': logits}

    def objective(self, x, y, epoch, indices=None):
        if self.method == 'MMDynamics':
            # One forward only: dropout cannot be redrawn for the auxiliary heads.
            captured = {}
            hooks = []
            for name in ('FeatureInforEncoder', 'TCPClassifierLayer', 'TCPConfidenceLayer'):
                for i, module in enumerate(getattr(self.network, name)):
                    hooks.append(module.register_forward_hook(
                        lambda _m, _args, value, key=(name, i): captured.__setitem__(key, value.detach())))
            try:
                loss, logits = self.network([x[:, i] for i in range(self.views)], y)
            finally:
                for hook in hooks: hook.remove()
            terms = {'author_total': loss.detach(), 'fusion_ce': F.cross_entropy(logits, y).detach()}
            for i in range(self.views):
                single = captured[('TCPClassifierLayer', i)]
                target = single.softmax(-1).gather(1, y[:, None]).flatten()
                terms[f'view{i}_gate_penalty'] = captured[('FeatureInforEncoder', i)].sigmoid().mean()
                terms[f'view{i}_single_ce'] = F.cross_entropy(single, y)
                terms[f'view{i}_confidence_mse'] = F.mse_loss(captured[('TCPConfidenceLayer', i)].flatten(), target)
            return loss, terms
        out = self(x)
        if self.method == 'CCML':
            loss = ccml.ccml_loss(out['evidence'], y, epoch, 50, 1.25, 1., 1.)
            with torch.no_grad():
                e = out['evidence'].detach()
                fused = ccml.evidence_dc(e, 1.25)
                vd, vr = ccml._loss_terms(e, y, epoch, 50)
                cd, cr = ccml._loss_terms(fused['e_con'], y, epoch, 50)
                complementary = ccml.expected_cross_entropy(fused['e_cmp'] + 1, F.one_hot(y, 2).to(e.dtype))
            return loss, {'ccml_full': loss.detach(), 'single_data': vd.sum(1).mean(),
                          'single_kl': vr.sum(1).mean(), 'consistent_data': cd.mean(),
                          'consistent_kl': cr.mean(), 'complement_data': complementary.mean()}
        if self.method == 'LMF':
            loss = F.cross_entropy(out['logits'], y)
            return loss, {'fusion_ce': loss.detach()}
        if self.method == 'ECML':
            e = out['evidence']
            onehot = F.one_hot(y, 2).to(e.dtype)
            def edl(value):
                alpha = value + 1
                return (ccml.expected_cross_entropy(alpha, onehot) + min(1., epoch/50) *
                        ccml.kl_to_uniform(1 + (1-onehot)*value)).mean()
            data = (edl(out['final']) + sum(edl(e[:, i]) for i in range(self.views))) / (self.views+1)
            alpha = e + 1
            p = alpha / alpha.sum(-1, keepdim=True)
            u = 2 / alpha.sum(-1)
            conflict = e.new_zeros(())
            for i in range(self.views):
                distance = (p - p[:, i:i+1]).abs().sum(-1)/2/(self.views-1)
                reliability = (1-u[:, i:i+1]) * (1-u)
                conflict = conflict + (distance * reliability).sum(1).mean()
            return data + conflict, {'edl_average': data.detach(), 'conflict_consistency': conflict.detach()}
        # QMF author RGBD loss: CE sum and lambda=.1 rank loss before history update.
        if indices is None:
            raise ValueError('QMF needs train-only local sample indices')
        ii = indices.detach().cpu().numpy()
        ce = F.cross_entropy(out['logits'], y)
        ranking = x.new_zeros(())
        per_sample = []
        for v in range(self.views):
            indiv = F.cross_entropy(out['logits_v'][:, v], y, reduction='none')
            ce = ce + indiv.mean()
            history = self.history[v]
            span = float(history.max() - history.min())
            # Author first batch history is all zero: define ties rather than NaN.
            scores = (history[ii] - history.min()) / span if span > 0 else np.zeros(len(ii))
            diff = scores - np.roll(scores, -1)
            target = torch.as_tensor(-np.sign(diff), device=x.device, dtype=x.dtype)
            margin = torch.as_tensor(np.abs(diff), device=x.device, dtype=x.dtype)
            confidence = out['quality'][:, v]
            ranking = ranking + torch.clamp(-target*(confidence-torch.roll(confidence, -1)) + margin, min=0).mean()
            per_sample.append(indiv.detach().cpu().numpy())
        quality = out['quality'].detach().cpu().numpy()
        for v in range(self.views):
            self.history[v, ii] += per_sample[v]
            self.confidence_history[v, ii] = quality[:, v]
        loss = ce + .1*ranking
        return loss, {'classification_sum': ce.detach(), 'ranking': ranking.detach()}
