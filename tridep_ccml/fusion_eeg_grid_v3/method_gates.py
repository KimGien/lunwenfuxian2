"""CPU engineering gates; no formal identities, GPU work or test predictions.

The QMF oracle is the registered upstream source, executed from its AST.  Only
Tensor.cuda() is removed for CPU execution; equal-history/singleton adaptations
are identified separately instead of being concealed in ordinary parity tests.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import time
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tridep_ccml.fusion_eeg_grid_v2.models import Fusion as OldFusion
from .models import Fusion, qmf_rank_loss
from .preprocessing import fit_preprocessor, apply_preprocessor, load_inputs

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'output/analysis/fusion_eeg15_methods5_v3_20261014/author_originals/QMF/RGBD-scene-recognition'
REPORT = ROOT / 'output/analysis/fusion_eeg15_methods5_v3_20261014/method_gates'
ATOL = RTOL = 2e-6


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def binding(path):
    path = Path(path)
    return {'path': path.relative_to(ROOT).as_posix(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


class CPUOnly(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
            require(not node.args and not node.keywords, 'Unexpected upstream cuda call')
            return ast.copy_location(node.func.value, node)
        return node


def source_node(path, name, parent=None):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    body = tree.body if parent is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == parent).body
    return next(n for n in body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)


def oracle():
    history_path = UPSTREAM / 'utils/crl_utils.py'
    train_path = UPSTREAM / 'train_QMF_nyud2.py'
    model_path = UPSTREAM / 'models/QMF.py'
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'np': np}
    records = []
    for path, name, parent in [(history_path, 'History', None), (train_path, 'rank_loss', None),
                               (train_path, 'model_forward_train', None), (model_path, 'forward', 'QMF')]:
        node = source_node(path, name, parent)
        records.append(dict(binding(path), symbol=name, line=node.lineno, end_line=node.end_lineno,
                            transform='remove no-argument Tensor.cuda() for CPU only'))
        tree = CPUOnly().visit(ast.Module(body=[node], type_ignores=[]))
        exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    RawHistory = namespace['History']

    class GuardedHistory(RawHistory):
        def correctness_normalize(self, data):
            # Explicit local guard; ordinary nonzero-history cases call the
            # original upstream function without changing its arithmetic.
            if self.correctness.max() == self.correctness.min():
                return np.zeros_like(data)
            return super().correctness_normalize(data)

    def author_rank(confidence, indices, history, guarded=False):
        h = (GuardedHistory if guarded else RawHistory)(len(history))
        h.correctness = history.copy()
        return namespace['rank_loss'](confidence, indices, h)

    return namespace, RawHistory, GuardedHistory, author_rank, records


def compare(a, b, name):
    require(a.shape == b.shape, name + ': shape mismatch')
    require(torch.allclose(a, b, atol=ATOL, rtol=RTOL), name + ': tolerance exceeded')
    return float((a - b).detach().abs().max()) if a.numel() else 0.


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def ranking_cases(author_rank):
    results = []
    histories = {
        'unequal': np.array([.8, .2, .0, 1., .1, .9, .4, .6, .25, .75, .5]),
        'partially_repeated': np.array([.8, .2, .2, 1., .8, 1., .4, .4, .0, .0, .5]),
        'all_equal': np.ones(11) * .73,
    }
    for views in (2, 3):
        for name, values in histories.items():
            for ids in (list(range(8)), [8, 9, 10]):
                for direction in ('correct', 'reversed'):
                    indices = torch.tensor(ids)
                    total_local, total_author = torch.zeros(()), torch.zeros(())
                    local_q, author_q = [], []
                    for v in range(views):
                        history = values + .17 * v
                        quality = torch.tensor(history[ids], dtype=torch.float32) * 3.
                        if direction == 'correct':
                            quality = -quality
                        qa = quality.clone().requires_grad_()
                        qb = quality.clone().requires_grad_()
                        local_q.append(qa); author_q.append(qb)
                        total_local = total_local + qmf_rank_loss(qa, indices, history, 'Q1')
                        total_author = total_author + author_rank(qb, indices, history, guarded=name == 'all_equal')
                    total_local.backward(); total_author.backward()
                    value_error = compare(total_local, total_author, 'author rank value')
                    grad_error = max(compare(a.grad, b.grad, 'author rank gradient') for a, b in zip(local_q, author_q))
                    results.append({'views': views, 'history_case': name, 'direction': direction,
                                    'batch_size': len(ids), 'value': total_local.item(),
                                    'max_value_error': value_error, 'max_gradient_error': grad_error,
                                    'explicit_equal_history_guard': name == 'all_equal'})
    # Raw author's globally equal case is undefined, and singleton squeeze
    # fails.  Keep these as documented local extensions, not author equivalence.
    with np.errstate(invalid='ignore', divide='ignore'):
        raw_equal = author_rank(torch.tensor([.1, .2], requires_grad=True), torch.tensor([0, 1]), np.ones(3))
    require(not torch.isfinite(raw_equal), 'Expected upstream equal-history undefined result')
    q = torch.tensor([.7], requires_grad=True)
    singleton = qmf_rank_loss(q, torch.tensor([1]), np.array([.1, .2, .3]), 'Q1')
    singleton.backward()
    require(singleton.item() == 0. and q.grad.item() == 0., 'Singleton guard differs')
    try:
        author_rank(q.detach(), torch.tensor([1]), np.array([.1, .2, .3]))
    except AssertionError:
        raw_singleton_asserts = True
    else:
        raw_singleton_asserts = False
    require(raw_singleton_asserts, 'Upstream singleton boundary changed')
    require(any(r['value'] > 0 for r in results if r['direction'] == 'reversed'), 'Missing active rank gradient cases')
    return {'cases': results, 'raw_author_all_equal_nonfinite': True,
            'raw_author_singleton_asserts': raw_singleton_asserts,
            'local_singleton_zero_value_and_gradient': True}


def author_logits(namespace, logits_v):
    # Invoke the upstream forward function with identity encoders/classifiers.
    # Inputs here are already real trainable logits, never log(probability).
    adapter = types.SimpleNamespace(rgbenc=nn.Identity(), depthenc=nn.Identity(),
                                    clf_rgb=[], clf_depth=[])
    forward = namespace['forward']
    pair = forward(adapter, logits_v[:, 0], logits_v[:, 1])
    fused, _, _, q0, q1 = pair
    qualities = [q0.reshape(-1), q1.reshape(-1)]
    if logits_v.shape[1] == 3:
        third = forward(adapter, logits_v[:, 2], logits_v[:, 2])
        q2 = third[3].reshape(-1)
        qualities.append(q2)
        fused = fused + logits_v[:, 2] * q2[:, None]
    return fused, torch.stack(qualities, 1)


def full_objective_cases(namespace, History):
    rows = []
    for views in (2, 3):
        torch.manual_seed(901 + views)
        a = Fusion('QMF', views, 11, 'Q1')
        b = copy.deepcopy(a)
        histories = [History(11) for _ in range(views)]
        for v, h in enumerate(histories):
            h.correctness = np.linspace(0., 3., 11) + .1 * v
            a.history[v] = h.correctness.copy()
        x = torch.randn(11, views, 128)
        y = torch.arange(11) % 2
        opta = torch.optim.AdamW(a.parameters(), lr=1e-3, weight_decay=1e-3, foreach=False)
        optb = torch.optim.AdamW(b.parameters(), lr=1e-3, weight_decay=1e-3, foreach=False)
        for step, ids in enumerate((list(range(8)), [8, 9, 10], [2, 0, 8, 6, 4, 1, 10, 3], [5, 7, 9])):
            indices = torch.tensor(ids)
            rng = torch.get_rng_state().clone()
            opta.zero_grad(); optb.zero_grad()
            loss, _ = a.objective(x[indices], y[indices], step + 1, indices)
            loss.backward()
            torch.set_rng_state(rng)
            logits_v = torch.stack([b.heads[v](x[indices, v]) for v in range(views)], 1)
            fused, quality = author_logits(namespace, logits_v)
            if views == 2:
                # Execute the entire author's training objective with an adapter
                # returning the unchanged upstream forward tensors.
                def model(rgb, depth):
                    return fused, logits_v[:, 0], logits_v[:, 1], quality[:, :1], quality[:, 1:2]
                batch = {'A': x[indices, 0], 'B': x[indices, 1], 'label': y[indices], 'idx': indices}
                ref = namespace['model_forward_train'](step + 1, model, types.SimpleNamespace(lamb=.1),
                                                         batch, histories[1], histories[0])[0]
            else:
                # The pinned source is two-modality; documented 3-view extension
                # repeats each actual upstream rank/history mechanism once.
                ce = F.cross_entropy(fused, y[indices])
                rank = fused.new_zeros(())
                indiv = []
                for v in range(views):
                    item = F.cross_entropy(logits_v[:, v], y[indices], reduction='none')
                    indiv.append(item.detach())
                    ce = ce + item.mean()
                    rank = rank + namespace['rank_loss'](quality[:, v], indices, histories[v])
                for v in range(views):
                    histories[v].correctness_update(indices, indiv[v], quality[:, v])
                ref = ce + .1 * rank
            ref.backward()
            ve = compare(loss, ref, 'full objective')
            ge = max(compare(p.grad, q.grad, 'full parameter gradient') for p, q in zip(a.parameters(), b.parameters()))
            for v in range(views):
                require(np.allclose(a.history[v], histories[v].correctness, atol=ATOL, rtol=RTOL), 'Cumulative train history differs')
                require(np.allclose(a.confidence_history[v], histories[v].confidence, atol=ATOL, rtol=RTOL), 'Quality history differs')
            torch.nn.utils.clip_grad_norm_(a.parameters(), 1., error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(b.parameters(), 1., error_if_nonfinite=True)
            opta.step(); optb.step()
            pe = max(compare(p, q, 'updated parameter') for p, q in zip(a.parameters(), b.parameters()))
            rows.append({'views': views, 'step': step + 1, 'batch_size': len(ids),
                         'value_error': ve, 'max_parameter_gradient_error': ge,
                         'max_updated_parameter_error': pe, 'history_passed': True})
    return rows


def restore_cases(output):
    rows = []
    for views in (2, 3):
        for variant in ('Q0', 'Q1'):
            torch.manual_seed(270 + views)
            x, y = torch.randn(11, views, 128), torch.arange(11) % 2
            model = Fusion('QMF', views, 11, variant)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3, foreach=False)

            def advance(model, optimizer, start, stop):
                records = []
                for epoch in range(start, stop):
                    order = torch.randperm(len(y))
                    losses = []
                    for offset in range(0, len(order), 8):
                        ids = order[offset:offset + 8]
                        optimizer.zero_grad(set_to_none=True)
                        loss, _ = model.objective(x[ids], y[ids], epoch, ids)
                        require(torch.isfinite(loss), 'Nonfinite continuation loss')
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step(); losses.append(loss.item())
                    records.append({'epoch': epoch, 'order': order.tolist(), 'losses': losses})
                return records

            prefix = advance(model, optimizer, 1, 3)
            saved = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'history': model.history.copy(), 'confidence_history': model.confidence_history.copy(),
                     'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
                     'next_epoch': 3, 'sampling_boundary': 'whole_epoch', 'qmf_variant': variant,
                     'prefix': prefix}
            path = output / f'engineering_resume_{views}_{variant}.pt'
            torch.save(saved, path)
            expected_records = advance(model, optimizer, 3, 5)
            restored = Fusion('QMF', views, 11, variant)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3, weight_decay=1e-3, foreach=False)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            restored.load_state_dict(checkpoint['model'])
            restored_optimizer.load_state_dict(checkpoint['optimizer'])
            restored.history = checkpoint['history'].copy()
            restored.confidence_history = checkpoint['confidence_history'].copy()
            torch.set_rng_state(checkpoint['torch_rng']); np.random.set_state(checkpoint['numpy_rng'])
            actual_records = advance(restored, restored_optimizer, checkpoint['next_epoch'], 5)
            require(equal(expected_records, actual_records) and equal(model.state_dict(), restored.state_dict())
                    and equal(optimizer.state_dict(), restored_optimizer.state_dict())
                    and equal(model.history, restored.history) and equal(model.confidence_history, restored.confidence_history),
                    'Continuous/saved-resumed CPU states are not bitwise equal')
            history_before = model.history.copy()
            model.eval()
            with torch.inference_mode():
                model(x)
            require(np.array_equal(history_before, model.history), 'Inference changed train history')
            rows.append({'views': views, 'variant': variant, 'continuous_and_resume_bitwise': True,
                         'epochs': 4, 'updates_continuous': 8, 'updates_resume_replay': 4,
                         'batch_sizes': [8, 3], 'inference_history_unchanged': True, 'checkpoint': binding(path)})
    return rows


def preprocessing_cases():
    rng = np.random.default_rng(20261014)
    train = rng.normal(size=(27, 3, 128)).astype(np.float32)
    train[:, :, 0] = 0.; train[:, :, 1] = 5.
    train[:, :, 2] *= 1e-8
    unseen = rng.normal(size=(10, 3, 128)).astype(np.float32)
    unseen[:, :, 0] = 100.; unseen[:, :, 1] = -200.; unseen[:, :, 2] = 300.
    state = fit_preprocessor(train, 'N1', [f'p{i:02}' for i in range(27)])
    same_state = fit_preprocessor(train.copy(), 'N1', state['fit_people'])
    transformed = apply_preprocessor(np.concatenate([train, unseen]), state)
    mask = np.asarray(state['mask'])
    require(np.all(transformed[:, mask] == 0.), 'Constant/near-constant unseen coordinates were not masked')
    require(equal(state, same_state), 'Training statistics depend on unseen inputs')
    old_mean = train.astype(np.float64).mean(0); old_std = train.astype(np.float64).std(0)
    expected = ((np.concatenate([train, unseen]).astype(np.float64) - old_mean) / np.maximum(old_std, 1e-6)).astype(np.float32)
    require(np.array_equal(transformed[:, ~mask], expected[:, ~mask]), 'Unmasked coordinates differ')
    n0 = apply_preprocessor(np.concatenate([train, unseen]), fit_preprocessor(train, 'N0'))
    require(np.array_equal(n0, expected), 'N0 does not reproduce actual old algorithm')
    require(np.array_equal(transformed, apply_preprocessor(np.concatenate([train, unseen]), json.loads(json.dumps(state)))),
            'Normalization serialization changes output')
    mutated = unseen.copy(); mutated *= -1200.
    apply_preprocessor(np.concatenate([train, mutated]), state)
    require(equal(state, same_state), 'Application mutated training statistics')
    # Boundary tested with exactly representable float64 +/- threshold.
    boundary = np.tile(np.array([-1e-6, 1e-6])[:, None, None], (1, 2, 128))
    boundary[:, :, 1] *= 1.01
    bs = fit_preprocessor(boundary, 'N1')
    require(bs['mask'][0][0] and not bs['mask'][0][1], 'Threshold must use <= 1e-6')
    actual_rows = []
    for eeg, seed, fold in [('ST', 33, 1), ('E00', 11, 1), ('E10', 22, 2), ('E15', 33, 3)]:
        identity = {'eeg': eeg, 'seed': seed, 'fold': fold}
        x, y, ix, binding_data = load_inputs(identity, 'cpu', 'N1')
        local_mask = np.asarray(binding_data['normalization']['mask'])
        require(np.all(x.numpy()[:, local_mask] == 0.), 'Actual cache mask not applied')
        require(len(set(binding_data['subject_ids'])) == 37 and int(y.sum()) == 16, 'Actual role/label binding differs')
        actual_rows.append({'identity': identity, 'normalization': binding_data['normalization'],
                           'sources': binding_data['sources'], 'roles': binding_data['roles'],
                           'input_distributions': binding_data['input_distributions']})
    return {'synthetic_masked_dimensions': int(mask.sum()), 'unseen_activation_forced_zero': True,
            'unmasked_matches_original_bitwise': True, 'N0_actual_algorithm_bitwise': True,
            'train_only_and_serialization_unchanged': True, 'threshold_boundary_passed': True,
            'synthetic_N0_abs_max': float(np.abs(n0).max()),
            'synthetic_N1_abs_max': float(np.abs(transformed).max()),
            'actual_input_checks': actual_rows}


def preserved_methods():
    records = []
    for views in (2, 3):
        for method in ('CCML', 'LMF', 'MMDynamics', 'QMF', 'ECML'):
            torch.manual_seed(113)
            old = OldFusion(method, views, 11)
            new = Fusion(method, views, 11, 'Q0')
            new.load_state_dict(old.state_dict())
            if method == 'QMF':
                old.history[:] = np.linspace(0., 1., 11)
                new.history[:] = old.history
            x = torch.randn(8, views, 128); y = torch.arange(8) % 2
            rng = torch.get_rng_state().clone()
            lo, terms_old = old.objective(x, y, 13, torch.arange(8)); lo.backward()
            torch.set_rng_state(rng)
            ln, terms_new = new.objective(x, y, 13, torch.arange(8)); ln.backward()
            require(torch.equal(lo, ln) and equal(terms_old, terms_new), 'Preserved method objective differs')
            require(all(torch.equal(a.grad, b.grad) for a, b in zip(old.parameters(), new.parameters())),
                    'Preserved method gradients differ')
            require(equal(old.history, new.history) and equal(old.confidence_history, new.confidence_history),
                    'Legacy quality history changed')
            # Actual N1 inputs, synthetic labels only, finite full gradients and
            # parameter updates for every mechanism at both modality counts.
            raw = torch.randn(11, views, 128).numpy(); raw[:, :, 0] = 0.
            normalized = torch.from_numpy(apply_preprocessor(raw, fit_preprocessor(raw, 'N1')))
            model = Fusion(method, views, 11, 'Q1')
            initial = copy.deepcopy(model.state_dict())
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3, foreach=False)
            components = set()
            for ids in (torch.arange(8), torch.arange(8, 11)):
                optimizer.zero_grad(set_to_none=True)
                loss, terms = model.objective(normalized[ids], ids % 2, 1, ids)
                require(torch.isfinite(loss) and all(torch.isfinite(t) for t in terms.values()), 'Nonfinite complete loss')
                loss.backward()
                require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()), 'Missing/nonfinite gradient')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); components.update(terms)
            require(not equal(initial, model.state_dict()), 'No parameter update')
            records.append({'method': method, 'views': views, 'v2_Q0_preservation_bitwise': True,
                            'N1_input_loss_gradient_parameter_finite': True, 'actual_updates': 2,
                            'parameter_changed': True, 'components': sorted(components)})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=REPORT / f'run_{time.time_ns()}')
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(REPORT.resolve()), 'Engineering output outside registered directory')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    namespace, RawHistory, GuardedHistory, rank, sources = oracle()
    report = {'schema': 'fusion_v3_method_gates_1', 'passed': False, 'device': 'cpu',
              'precision': 'float32 model; float64 cumulative history and normalization',
              'author_commit': 'fe6c4c6ef7cb23f0a89594ee413d485f1854268b',
              'tolerance_preregistered': {'atol': ATOL, 'rtol': RTOL}, 'author_sources': sources,
              'ranking': ranking_cases(rank),
              'full_author_objective_and_updates': full_objective_cases(namespace, GuardedHistory),
              'checkpoint_continuation': restore_cases(output),
              'normalization': preprocessing_cases(), 'complete_methods': preserved_methods(),
              'source_differences': {
                  'Q1': 'Pinned RGBD source: max(0, sign(delta normalized cumulative CE)*delta quality - abs(delta normalized cumulative CE)).',
                  'Q0': 'Actual legacy variant: max(0, sign(delta normalized cumulative CE)*delta quality + abs(delta normalized cumulative CE)).',
                  'printed_eq16_17': 'max(0, sign(delta quality)*delta average CE + abs(delta quality)); sign and margin apply to quality, unlike author code.',
                  'history': 'Paper Eq14 uses temporal average; bound code accumulates per-sample CE then min/max normalizes across train history. Rank is evaluated before current batch CE is accumulated.',
                  'equal_history': 'Raw upstream normalization divides 0/0. Local N0/Q0 and Q1 define global equal history as zero normalized scores and tied zero rank loss/gradient.',
                  'three_modalities': 'Author RGBD path has two views; registered EST extension sums all three view CE/rank terms and quality-weighted real logits in E,S,T order.',
                  'singleton': 'Raw upstream squeeze triggers assertion; local shape-safe self-pair gives exactly zero loss and gradient.',
                  'readout': 'Original image encoders are replaced by registered frozen 128-dimensional representations and fixed 128->32->2 heads; all five mechanisms retain registered v2 capacity.'},
              'scope': {'formal_fits': 0, 'gpu_operations': 0, 'candidate_test_predictions': 0,
                        'sealed_sources_modified': False, 'source_cache_reads_are_input_distribution_only': True},
              'local_sources': [binding(Path(__file__)), binding(Path(__file__).with_name('models.py')),
                                binding(Path(__file__).with_name('preprocessing.py')),
                                binding(ROOT / 'output/research/fusion_candidates_20261008/QMF_fulltext.txt'),
                                binding(ROOT / 'output/analysis/fusion_eeg15_methods5_v3_20261014/author_source_comparison.json')],
              'torch_version': torch.__version__}
    report['engineering_seconds'] = time.perf_counter() - start
    report['passed'] = True
    path = output / 'method_gate_report.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'report': str(path), 'seconds': report['engineering_seconds'],
                      'ranking_cases': len(report['ranking']['cases']),
                      'full_objective_steps': len(report['full_author_objective_and_updates'])}))


if __name__ == '__main__':
    main()
