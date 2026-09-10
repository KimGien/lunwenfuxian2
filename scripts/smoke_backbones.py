"""Load only locally pinned backbone weights; no remote model code/network."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import json
import torch
from transformers import BertModel, BertTokenizer, HubertModel, Wav2Vec2FeatureExtractor
from prepare import ROOT,save,digest

def main():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    report={'device':device,'synthetic_inputs_only':True}
    path=ROOT/'models/macbert'
    model,info=BertModel.from_pretrained(path,local_files_only=True,output_loading_info=True)
    if info['missing_keys'] or info['mismatched_keys']: raise ValueError(info)
    model.to(device).eval(); tokenizer=BertTokenizer.from_pretrained(path,local_files_only=True)
    inputs=tokenizer(['这是用于检查模型输入的测试句子。'],return_tensors='pt').to(device)
    with torch.no_grad(): x=model(**inputs).last_hidden_state[:,0,:]
    assert x.shape==(1,768) and torch.isfinite(x).all()
    report['macbert']={'loading_info':info,'shape':list(x.shape),'weight_sha256':digest(path/'pytorch_model.bin')}
    del model
    path=ROOT/'models/hubert_code'
    model,info=HubertModel.from_pretrained(path,local_files_only=True,output_loading_info=True)
    if info['missing_keys'] or info['mismatched_keys']: raise ValueError(info)
    model.to(device).eval()
    with torch.no_grad(): x=model(torch.zeros(1,80000,device=device)).last_hidden_state
    assert x.shape[-1]==768 and torch.isfinite(x).all()
    report['hubert_code']={'loading_info':info,'frame_shape':list(x.shape),'pooled_dimension':int(x.shape[-1]),'weight_sha256':digest(path/'pytorch_model.bin')}
    save(ROOT/'reports/backbone_smoke.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
