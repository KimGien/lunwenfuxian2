"""Fixed independent AS-R and AS-P contrasts; all previous artifacts remain immutable."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import time

from . import as_projection as ap
from . import short_train as ast
from . import short_audio as asp
from . import nr
from .common import WORK, TC, RecordContext, read_json, write_new, descriptor, verify_descriptor, digest, require, set_seed, LOSS
from .rebuild_eeg import predict_rows, validate_row, loss_parts
from .complete_eeg import mean_scores, crosscheck_metrics
from .complete_nr import diagnostics
from .complete_p import gate
from tridep_ccml.train import make_initial_state, state_digest
from tridep_ccml.ccml import ccml_loss, evidence_dc
from tridep_ccml.metrics import summarize
import numpy as np
import torch
from torch.nn import functional as F

OUT = TC/'experiments/CCML-v2-AS-RP-37-fulltext'
CONFIG = OUT/'configuration.json'
PLAN = WORK/'output/plans/CCML_v2_AS_R_P_补充固定对照计划.md'
VARIANTS = {'R': {'experiment': 'CCML-v2-AS-R-37-fulltext', 'parameters': 774, 'weight_decay': 1e-4},
            'P': {'experiment': 'CCML-v2-AS-P-37-fulltext', 'parameters': 476294, 'weight_decay': 0.}}
MODALITIES = ap.MODALITIES


def protect_history():
    path = OUT/'history_before.json'
    if not path.exists():
        source = TC/'experiments/CCML-v2-S-37-fulltext'
        frozen = read_json(source/'history_before.json')
        for p in [*source.rglob('*'), TC/'v2/scale.py', TC/'v2/complete_scale.py', TC/'v2/test_scale.py',
                  WORK/'output/plans/CCML_v2_S_模态证据尺度校准固定计划.md',
                  *list((TC/'experiments').glob('CCML-v2-S-*.log'))]:
            if p.is_file():
                frozen[p.relative_to(WORK).as_posix()] = digest(p)
        write_new(path, frozen)
    frozen = read_json(path)
    for name, sha in frozen.items():
        require(digest(WORK/name) == sha, 'Historical artifact changed: '+name)
    return len(frozen)


def configuration(context):
    paths = [Path(__file__), TC/'v2/test_as_optional.py', TC/'v2/nr.py', TC/'v2/complete_nr.py',
             TC/'v2/complete_p.py', TC/'v2/complete_eeg.py', *[verify_descriptor(d) for d in read_json(ap.CONFIG)['sources'].values()]]
    return {'family': OUT.name, 'date': '2026-09-24', 'plan': descriptor(PLAN), 'variants': VARIANTS,
            'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256,
            'subject_ids': context.ids, 'seeds': [11,22,33], 'folds': [1,2,3,4,5],
            'comparator': asp.STAGES['control'], 'parent_status': descriptor(asp.OUT/'status.json'),
            'AS_NP_status': descriptor(ap.OUT/'status.json'), 'normalization': 'none', 'dropout': 'eval',
            'optimizer': 'Adamax', 'head_lr': .001, 'projection_lr': .0001, 'gradient_clip': 1.,
            'epochs': 60, 'batch_size': 8, 'ccml': LOSS, 'selection': 'fixed_last_epoch',
            'threshold': .5, 'tie_class': 0, 'device': 'cpu', 'threads': 1, 'search': False,
            'runs': 30, 'updates': 7200, 'sources': [descriptor(p) for p in sorted(set(paths))]}


def make_model(data, seed, variant):
    if variant == 'P':
        return ap.NormalizedProjectionHeads(data['initial'], seed, normalize=False)
    require(variant == 'R', 'Unknown variant')
    model = nr.CandidateHeads('R')
    model.load_state_dict(make_initial_state(seed), strict=True)
    return model.eval()


def optimizer_for(model, variant):
    groups = [{'params': model.projections.parameters(), 'lr': .0001},
              {'params': model.heads.parameters(), 'lr': .001}] if variant == 'P' else [{'params': model.parameters(), 'lr': .001}]
    return torch.optim.Adamax(groups, weight_decay=VARIANTS[variant]['weight_decay'])


def load_data(context, seed, fold):
    p = ap.load_data(context, seed, fold)
    r = ast.load_data(context, seed, fold, 'control')
    return {'P': p, 'R': r}


def rows_for(model, data, context, seed, fold, variant, cp):
    logits = ap.logits_all(model, data) if variant == 'P' else nr.logits_all(model, data['features'])
    rows = predict_rows(logits, data, context, seed, fold, VARIANTS[variant]['experiment'], descriptor(cp), data['source'])
    for row in rows:
        row['configuration'] = descriptor(CONFIG)
        validate_row(row)
    return rows


def baseline_run(seed, fold):
    return asp.OUT/'runs'/asp.STAGES['control']/f'seed_{seed}/fold_{fold}'


def preflight():
    OUT.mkdir(parents=True, exist_ok=True)
    context = RecordContext()
    count = protect_history()
    require(read_json(OUT/'test_results.json')['passed'], 'Tests missing')
    write_new(CONFIG, configuration(context))
    checks = []
    for seed in [11,22,33]:
        for fold in range(1,6):
            set_seed(seed)
            data = load_data(context, seed, fold)
            ap.verify_initial_parents(context, seed, fold, data['P'])
            base = baseline_run(seed, fold)
            for d in read_json(base/'complete.json').values(): verify_descriptor(d)
            saved = torch.load(base/'checkpoint.pt', map_location='cpu', weights_only=True)
            meta = saved['metadata']
            r, p = (make_model(data[v], seed, v) for v in ['R','P'])
            for model, v in [(r,'R'),(p,'P')]:
                require(sum(x.numel() for x in model.parameters()) == VARIANTS[v]['parameters'], 'Parameter scope differs')
            require(meta['head_initialization_sha256'] == state_digest(make_initial_state(seed)), 'Fresh head pairing differs')
            r.load_state_dict(saved['state_dict'], strict=True)
            p.heads.load_state_dict(saved['state_dict'], strict=True)
            expected = read_json(base/'predictions.json')
            rp = evidence_dc(F.softplus(nr.logits_all(r, data['R']['features'])))['p'].numpy()
            pp = evidence_dc(F.softplus(ap.logits_all(p, data['P'])))['p'].numpy()
            require(np.array_equal(rp, np.array([x['p_final'] for x in expected])), 'Frozen control replay not exact')
            require(np.allclose(ap.original_projection.project_all(p, data['P']).numpy(), data['R']['features'], atol=2e-5, rtol=2e-5), 'Projection reconstruction differs')
            error = float(np.max(np.abs(pp-rp)))
            require(error <= 2e-6 and (pp[:,1]>.5).tolist() == [bool(x['prediction']) for x in expected], 'Projected control replay differs')
            checks.append({'seed': seed, 'fold': fold, 'control_records_exact': 37,
                           'projected_control_max_probability_error': error, 'source': {k: d['source'] for k,d in data.items()}})
            print(f'AS-R/P preflight seed={seed} fold={fold} matched.', flush=True)
    write_new(OUT/'preflight.json', {'passed': True, 'configuration': descriptor(CONFIG), 'checks': checks,
                                   'protected_files_unchanged': count, 'control_records_exact': 555})


def fit(context, data, seed, fold, variant):
    start = time.perf_counter()
    set_seed(seed)
    model = make_model(data, seed, variant)
    folder = OUT/f'runs/{variant}/seed_{seed}/fold_{fold}'
    folder.mkdir(parents=True, exist_ok=True)
    cp = folder/'checkpoint.pt'
    train_ids = context.folds[f'fold_{fold}']['train']
    meta = {'experiment': VARIANTS[variant]['experiment'], 'variant': variant, 'seed': seed, 'fold': fold,
            'train_ids': train_ids, 'source': data['source'], 'configuration': descriptor(CONFIG),
            'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256,
            'initial_state_sha256': state_digest(model.state_dict()), 'head_initialization_sha256': state_digest(make_initial_state(seed)),
            'epochs': 60, 'updates': 240, 'objective': 'full_CCML_only', 'hyperparameters': VARIANTS[variant],
            'normalization': 'none', 'dropout': 'eval', 'torch': str(torch.__version__)}
    if cp.exists():
        side = read_json(folder/'checkpoint.json')
        saved = torch.load(verify_descriptor(side['checkpoint']), map_location='cpu', weights_only=True)
        require(side['metadata'] == saved['metadata'] and all(saved['metadata'][k] == v for k,v in meta.items()), 'Resume provenance differs')
        model.load_state_dict(saved['state_dict'], strict=True)
    else:
        idx = [context.ids.index(s) for s in train_ids]
        require(not set(train_ids)&set(context.folds[f'fold_{fold}']['val']+context.folds[f'fold_{fold}']['test']), 'Training overlap')
        x = {m: torch.from_numpy(data[m][idx]) for m in MODALITIES} if variant=='P' else torch.from_numpy(data['features'][idx])
        y = torch.from_numpy(data['labels'][idx])
        opt = optimizer_for(model, variant)
        gen, batch_hash, history = torch.Generator().manual_seed(seed), hashlib.sha256(), []
        updates = 0
        for epoch in range(1,61):
            order = torch.randperm(len(idx), generator=gen)
            batch_hash.update(order.numpy().astype('<i8').tobytes())
            total, terms, norms = 0., {}, []
            for start_batch in range(0,len(idx),8):
                b=order[start_batch:start_batch+8]
                opt.zero_grad(set_to_none=True)
                e = F.softplus(model({m:v[b] for m,v in x.items()} if variant=='P' else x[b]))
                loss = ccml_loss(e,y[b],epoch,**LOSS)
                loss.backward()
                require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()), 'Invalid gradient')
                norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)))
                opt.step(); updates+=1
                require(all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite parameter')
                total += float(loss.detach())*len(b)
                with torch.no_grad():
                    for k,v in loss_parts(e.detach(),y[b],epoch).items(): terms[k]=terms.get(k,0.)+v*len(b)
            history.append({'epoch':epoch, 'train_loss':total/len(idx), 'loss_terms':{k:v/len(idx) for k,v in terms.items()},
                            'preclip_grad_norm_mean':float(np.mean(norms))})
        require(updates == 240, 'Update budget differs')
        meta['batch_order_sha256']=batch_hash.hexdigest()
        baseline=read_json(baseline_run(seed,fold)/'checkpoint.json')['metadata']
        require(meta['batch_order_sha256']==baseline['batch_order_sha256'] and meta['head_initialization_sha256']==baseline['head_initialization_sha256'], 'Control pairing differs')
        with cp.open('xb') as stream: torch.save({'state_dict':model.state_dict(),'metadata':meta},stream)
        write_new(folder/'checkpoint.json',{'checkpoint':descriptor(cp),'metadata':meta})
        write_new(folder/'train_history.json',history)
    write_new(folder/'predictions.json',rows_for(model,data,context,seed,fold,variant,cp))
    write_new(folder/'complete.json',{n:descriptor(folder/n) for n in ['checkpoint.pt','checkpoint.json','train_history.json','predictions.json']})
    print(f'AS-{variant} seed={seed} fold={fold} updates=240 seconds={time.perf_counter()-start:.1f}',flush=True)


def train():
    context=RecordContext()
    require(read_json(CONFIG)==configuration(context) and read_json(OUT/'preflight.json')['passed'], 'Locked configuration/preflight differs')
    for seed in [11,22,33]:
        for fold in range(1,6):
            data=load_data(context,seed,fold)
            for v in VARIANTS:fit(context,data[v],seed,fold,v)


def complete():
    context=RecordContext()
    require(read_json(CONFIG)==configuration(context), 'Configuration drift')
    rows={v:[] for v in VARIANTS}; baseline=[]; checks=[]
    for seed in [11,22,33]:
        for fold in range(1,6):
            data=load_data(context,seed,fold)
            baseline+=read_json(baseline_run(seed,fold)/'predictions.json')
            base_meta=read_json(baseline_run(seed,fold)/'checkpoint.json')['metadata']
            for v in VARIANTS:
                folder=OUT/f'runs/{v}/seed_{seed}/fold_{fold}'
                for d in read_json(folder/'complete.json').values():verify_descriptor(d)
                side=read_json(folder/'checkpoint.json'); cp=verify_descriptor(side['checkpoint'])
                saved=torch.load(cp,map_location='cpu',weights_only=True);meta=saved['metadata']
                require(meta==side['metadata'] and meta['configuration']==descriptor(CONFIG) and meta['source']==data[v]['source'], 'Source mismatch')
                require(meta['experiment']==VARIANTS[v]['experiment'] and meta['variant']==v and meta['seed']==seed and meta['fold']==fold and
                        meta['train_ids']==context.folds[f'fold_{fold}']['train'] and meta['epochs']==60 and meta['updates']==240 and meta['hyperparameters']==VARIANTS[v], 'Run metadata mismatch')
                set_seed(seed);model=make_model(data[v],seed,v)
                require(state_digest(model.state_dict())==meta['initial_state_sha256'], 'Initial state differs')
                if v=='P':
                    for m in MODALITIES:
                        keys=[k for k in model.state_dict() if k.startswith('projections.'+m)]
                        require(any(not torch.equal(model.state_dict()[k],saved['state_dict'][k]) for k in keys), 'Projection unchanged')
                model.load_state_dict(saved['state_dict'],strict=True)
                observed=read_json(folder/'predictions.json')
                require(observed==rows_for(model,data[v],context,seed,fold,v,cp), 'Exact replay differs')
                require(meta['batch_order_sha256']==base_meta['batch_order_sha256'] and meta['head_initialization_sha256']==base_meta['head_initialization_sha256'], 'Pairing mismatch')
                history=read_json(folder/'train_history.json')
                require([r['epoch'] for r in history]==list(range(1,61)) and all(math.isclose(r['train_loss'],sum(r['loss_terms'].values()),rel_tol=1e-5,abs_tol=1e-5) for r in history), 'Training history differs')
                rows[v]+=observed;checks.append({'variant':v,'seed':seed,'fold':fold,'exact_replayed_records':37,'checkpoint':descriptor(cp)})
    summaries={'control':summarize(baseline), **{v:summarize(x) for v,x in rows.items()}}
    for v in VARIANTS:
        require(len(rows[v])==555 and Counter((r['seed'],r['subject_id']) for r in rows[v] if r['split']=='test')==Counter((s,i) for s in [11,22,33] for i in context.ids),'OOF coverage differs')
    as_audit=read_json(asp.OUT/'completion_audit.json')
    require(summaries['control']['primary_seed_mean_of_fold_means']==as_audit['summaries']['AS']['control']['primary_seed_mean_of_fold_means'],'Control summary differs')
    n=as_audit['summaries']['AS']['N'];np_summary=read_json(ap.OUT/'completion_audit.json')['summaries']['candidate']
    result={'passed':True,'runs':30,'updates':7200,'exact_replayed_records':1110,'checks':checks,
            'summaries':summaries,'comparisons':{v:gate(summaries[v],summaries['control']) for v in VARIANTS},
            'normalization_context':{'N_vs_control':gate(n,summaries['control']),'NP_vs_P':gate(np_summary,summaries['P'])},
            'diagnostics':{v:diagnostics(x,baseline) for v,x in rows.items()},'control_diagnostics':diagnostics(baseline,baseline),
            'metric_crosschecks':sum(crosscheck_metrics(x) for x in rows.values()),'protected_files_unchanged':protect_history()}
    write_new(OUT/'completion_audit.json',result)
    lines=['# CCML-v2-AS-R / AS-P 补充固定对照完成报告','','日期：2026-09-24。两项独立对照均已完成，不改写历史实验。','',
           '| 条件 | Macro-F1 | MDD recall | NLL | Brier |','|---|---:|---:|---:|---:|']
    for k in ['control','R','P']:
        s=summaries[k]['primary_seed_mean_of_fold_means']
        lines.append('| AS-'+k+' | '+' | '.join(f'{s[m]:.6f}' for m in ['macro_f1','mdd_recall','nll','brier'])+' |')
    lines+=['','主口径：每种子五折均值→三种子均值。直接对照均为旧 AS 完整语音覆盖的 AS-control；与新 S 均值汇聚版本不是同输入比较。','']
    for v in VARIANTS:
        c=result['comparisons'][v];d=c['delta']
        lines.append(f'- AS-{v}：F1 {100*d["macro_f1"]:+.3f} 个百分点，{sum(x>0 for x in c["seed_f1_deltas"].values())}/3 种子提高；召回 {100*d["mdd_recall"]:+.3f} 个百分点；'+('通过' if c['gate']['development_criteria_passed'] else '未通过')+'预定开发门槛。')
    lines+=['','AS-R 仅增加头参数 Adamax weight_decay=1e-4（包括 bias）；AS-P 仅允许合法同折三路末端投影和新头更新，无 LayerNorm、无权重衰减。参数分别为 774 和 476,294。保持完整 CCML；固定 60 轮、batch 8、末轮、阈值 0.5，不搜索。','',
            '| 条件 | Train F1 | Val F1 | Test F1 |','|---|---:|---:|---:|']
    for v in ['control','R','P']:
        d=result['control_diagnostics'] if v=='control' else result['diagnostics'][v]
        lines.append('| AS-'+v+' | '+' | '.join(f'{d["train_val_test"][p]["final"]["macro_f1"]:.6f}' for p in ['train','val','test'])+' |')
    lines+=['','## 规范化的背景效应','',
            f'AS-N 相对 AS-control 的 F1 差为 {100*result["normalization_context"]["N_vs_control"]["delta"]["macro_f1"]:+.3f} 个百分点；AS-NP 相对 AS-P 为 {100*result["normalization_context"]["NP_vs_P"]["delta"]["macro_f1"]:+.3f} 个百分点。两项背景效应分别保留，不据此宣称统计显著交互，也不声称完整 N/R/P 因素矩阵已经完成。','',
            '## 验收与边界','',f'30 组、7,200 次更新完成。训练前 555 条 AS-control 预测精确复现；完成后 30 个检查点、1,110 条新预测精确复现，30 个外层测试折独立指标交叉核验通过。{read_json(OUT/"test_results.json")["tests_run"]} 项测试通过；{result["protected_files_unchanged"]:,} 项历史文件哈希不变。',
            '全部结果来自同一已查看的 37 人；历史同折父权重仍有监督学习血缘。本轮无内层选择、无新 CE、无新增人群验证。无论分数高低都保留，不自动更换默认或追加搜索。',
            '逐种子 OOF、ECE5、风险覆盖、证据强度、公共/互补贡献和分类变化保存于完整验收文件。','',
            '- [固定计划](../../../output/plans/CCML_v2_AS_R_P_补充固定对照计划.md)','- [验收与全部诊断](completion_audit.json)','- [固定配置](configuration.json)','- [完成状态](status.json)','']
    report=OUT/'CCML-v2-AS-RP实验完成报告.md';content='\n'.join(lines)
    if report.exists():require(report.read_text(encoding='utf8')==content,'Report changed')
    else:report.write_text(content,encoding='utf8')
    write_new(OUT/'status.json',{'status':'complete','date':'2026-09-24','runs':30,'updates':7200,'configuration':descriptor(CONFIG),'plan':descriptor(PLAN),
              'report':descriptor(report),'audit':descriptor(OUT/'completion_audit.json'),'passed_gates':{v:c['gate']['development_criteria_passed'] for v,c in result['comparisons'].items()}})
    print({k:v['primary_seed_mean_of_fold_means'] for k,v in summaries.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preflight','train','complete'])
    {'preflight':preflight,'train':train,'complete':complete}[parser.parse_args().action]()
