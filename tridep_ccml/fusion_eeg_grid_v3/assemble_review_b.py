"""Assemble independently reviewed, byte-bound startup evidence for B."""
from __future__ import annotations
import platform
import sys
import numpy as np
import torch
from . import common as C

def run():
    C.check_pause()
    evidence={
        'phase_a':C.REPORT/'phase_a/complete.json',
        'phase_a_independent':C.REPORT/'phase_a/independent_review.json',
        'source_gate':C.OUT/'source_gate.json',
        'method_gate':C.REPORT/'method_gates/run_1791945835937944100/method_gate_report.json',
        'source_binding_supplement':C.REPORT/'method_gates/source_binding_revision/gate_1791946276418683700.json',
        'training_gate':C.REPORT/'training_gates/run_1791946102292444900/complete.json',
        'parent_checkpoint_supplement':C.REPORT/'training_gates/parent_completeness_1791946345771857000.json',
        'training_independent':C.REPORT/'review_training_independent.json',
        'methods_independent':C.REPORT/'review_methods_independent.json',
        'queue_independent':C.REPORT/'review_queue_independent.json',
        'backend_gate':C.OUT/'backend_gate.json'
    }
    reports={name:C.read(path) for name,path in evidence.items()}
    for name,value in reports.items():
        C.require(value.get('passed') is True,'Startup evidence not passed: '+name)
    for name in ('training_independent','methods_independent','queue_independent'):
        for source in reports[name]['reviewed_sources']: C.checked(source)
    for source in reports['backend_gate']['sources']: C.checked(source)
    C.require(all(v['device']=='cpu' for v in reports['backend_gate']['methods'].values()),
              'Selected GPU method needs an additional same-device checkpoint recovery gate before approval')
    sources=[C.desc(C.CODE/name) for name in
             ('__init__.py','common.py','models.py','preprocessing.py','training.py','sources.py',
              'backend_gates.py','run_b.py','training_gates.py','method_gates.py','assemble_review_b.py')]
    sources += [C.desc(C.ROOT/name) for name in
                ('tridep_ccml/ccml.py','tridep_ccml/references/ccml/source.json',
                 'tridep_ccml/fusion_eeg_grid_v2/common.py','tridep_ccml/fusion_eeg_grid_v2/models.py',
                 'tridep_ccml/fusion_eeg_grid_v2/upstream_sources.json')]
    for item in reports['source_gate']['authors']:
        local=C.ROOT/item['local_path']
        C.require(C.sha(local)==item['local_sha256'],'Local vendor differs from author source acceptance')
        sources.append(C.desc(local))
        sources.append(item['original'])
        C.checked(item['original'])
    path=C.OUT/'implementation_review_b.json'
    report={'schema':'fusion_v3_independent_startup_review_B_1','passed':True,
            'formal_training_authorized':True,'scope':'B startup only; no claim of formal fits or final experiment acceptance',
            'reviewed_sources':sources,'evidence':{name:C.desc(p) for name,p in evidence.items()},
            'resolved_findings':['QMF nonzero-history objective now pinned author Q1, old Q0 isolated',
              'N1 masks only train-derived std<=1e-6; registered cache/metadata bytes enforced',
              'Completed fit reuse requires registration/config/device; no silent source change',
              'Milestone write interruption repaired from verified epoch-boundary latest',
              'D inherits nonzero patience and earliest minimum best; incomplete parents rejected',
              'Engineering backend uses same exclusive owner lease',
              'B completion certificate idempotent; post-owner failures released and recorded',
              'Original author Git blob verified; local LF/terminal newline differences disclosed'],
            'prior_gate_applicability':'Original full CPU recovery/method gates preserved with source snapshots. Final training differs only by stronger read-only parent checkpoint completeness checks, tested in parent supplement. Final preprocessing differs only by source byte verification, tested for bitwise outputs and rejection of individual/paired modifications. Independent reviews bind final versions.',
            'formal_accepted_backends':{k:v['device'] for k,v in reports['backend_gate']['methods'].items()},
            'environment':{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,
                           'platform':platform.platform(),'cuda_build':torch.version.cuda},
            'new_candidate_test_forwards_in_startup':0,'formal_fits_in_startup':0,'at':C.now()}
    C.save(path,report,immutable=True)
    print('Stage B startup independently accepted',flush=True)
    return report
if __name__=='__main__':run()

