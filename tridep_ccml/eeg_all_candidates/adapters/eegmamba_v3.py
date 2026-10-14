"""Unintegrated EEGMamba private-backward adapter; GPU qualification pending.

Only mixer-instance forward globals receive a private split-scan function.
The original native forward is retained; the private backward uses explicit
FP32 CPU prefix sums. Accumulation order changes and must be disclosed.
No installed module/class/global torch function is rewritten.
"""
import copy
import hashlib
import inspect
from pathlib import Path
import types
import torch
from . import eegmamba as V1
from .. import mamba_cumsum_cpu_fp32_v2 as C
from .. import mamba_private_backward_graph_v2 as G
from ._author_support import WORK,require

SOURCE=Path(__file__).resolve()
LIB=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/assets/eegmamba_linux_env/lib/python3.12/site-packages/mamba_ssm'
MIXER=LIB/'modules/mamba2.py'
COMBINED=LIB/'ops/triton/ssd_combined.py'
CHUNK=LIB/'ops/triton/ssd_chunk_state.py'
PINS={
 Path(V1.__file__):'1b7cef916818654108920efdd99873e0684614750bc9b32cdf6c25d1ca719d2a',
 Path(C.__file__):'689eec011f11cdf4f54a4cf2bc2545ac115011dded729f4048bc84de8381cabe',
 Path(G.__file__):'8686a7f890675b7026f0949be3f66f2ac607426bd0f89b1f0fc38829f74b33d8',
 MIXER:'605e4439ff0baec8d8acaf4a191d9f0570eea9900065a065909124c472b08707',
 COMBINED:'3717ba118605d9a5c9271d9195e2168c55b82242adcd46b143d3d9918f7c72ba',
 CHUNK:'c970c25ffcb5de3f0171dd035d73c0d59f62d711c0ebbc9a9c171c37194dc4be'}
FEATURE_DIM=V1.FEATURE_DIM
INPUT_CONTRACT=copy.deepcopy(V1.INPUT_CONTRACT)
INPUT_CONTRACT.update(adapter_version=3,backward_prefix_sum='private fixed left-to-right CPU FP32 for same-view helper out= and core new-result reverse scan; numerical order change declared')

def desc(p):
    p=Path(p).resolve()
    require(p.is_relative_to(WORK),'Source outside workspace')
    return {'path':p.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def check_sources():
    for path,digest in PINS.items():require(desc(path)['sha256']==digest,'Pinned Mamba source changed: '+str(path))

def class_factory(original,forward,backward):
    return type('PrivateFP32'+original.__name__,(torch.autograd.Function,),
                {'forward':staticmethod(forward),'backward':staticmethod(backward),'__module__':__name__})

def bind_mixer(mixer):
    check_sources()
    require(not getattr(mixer,'_private_fp32_scan_bound',False),'No double binding')
    original=mixer.forward.__func__
    require(type(mixer).__name__=='Mamba2' and Path(inspect.getsourcefile(original)).resolve()==MIXER.resolve()
            and mixer.use_mem_eff_path is True,'Pinned native Mamba2 memory-efficient path required')
    split=original.__globals__['mamba_split_conv1d_scan_combined']
    require(Path(inspect.getsourcefile(split)).resolve()==COMBINED.resolve(),'Split scan source differs')
    private,proof=G.build_private_split_scan(split,torch_namespace=C.torch_namespace(),class_factory=class_factory)
    new_forward=G.clone(original,{'mamba_split_conv1d_scan_combined':private})
    mixer.forward=types.MethodType(new_forward,mixer);mixer._private_fp32_scan_bound=True
    require(mixer.forward.__func__.__code__ is original.__code__
            and original.__globals__['mamba_split_conv1d_scan_combined'] is split,'Native mixer source/global changed')
    return proof

def load_encoder(device='cuda',trainable=False):
    check_sources()
    model,audit=V1.load_encoder(device=device,trainable=trainable)
    before=[n for n,_ in model.named_parameters()]
    selected=[n for n,p in model.named_parameters() if p.requires_grad]
    require(len(model.encoder.layers)==12,'Original12 Mamba2 blocks required')
    bindings=[bind_mixer(block.mixer) for block in model.encoder.layers]
    require(before==[n for n,_ in model.named_parameters()]
            and selected==[n for n,p in model.named_parameters() if p.requires_grad],
            'Parameter identities/trainability changed')
    audit.update(adapter_version=3,input_contract=copy.deepcopy(INPUT_CONTRACT),
                 runtime_backward_adaptation={'private_mixer_bindings':bindings,'prefix_source':desc(Path(C.__file__)),
                     'private_graph_source':desc(Path(G.__file__)),'accumulation_order_changed':True,
                     'core_reverse_cumsum_single_AST_call_changed':True,'original_core_bytecode_preserved':False,
                     'native_CUDA_numerical_equivalence_established':False,'host_transfer_cost_unmeasured':True,
                     'performance_improvement_claimed':False,'original_native_forward_function_reused':True},
                 GPU_qualification_complete=False,new_formal_execution_registration_required=True)
    audit['implementation_sources']=list(audit.get('implementation_sources',[]))+[desc(SOURCE)]+[desc(p) for p in PINS]
    return model,audit

def exact_finetune_parameter_names(model):return V1.exact_finetune_parameter_names(model)
def adapt_raw(raw):return V1.adapt_raw(raw)
def encode_windows(model,tensor):return V1.encode_windows(model,tensor)
