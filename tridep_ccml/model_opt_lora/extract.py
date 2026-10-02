from .core import *
from transformers import BertModel,BertTokenizer

def main():
    assert not (OUT/'registration.json').exists()
    ctx=RecordContext();(OUT/'cache/subjects').mkdir(parents=True,exist_ok=True)
    set_seed(11)
    tokenizer=BertTokenizer.from_pretrained(BACKBONE,local_files_only=True)
    model=BertModel.from_pretrained(BACKBONE,local_files_only=True,attn_implementation='eager').cuda().eval().requires_grad_(False)
    last={k:v.detach().cpu() for k,v in model.encoder.layer[-1].state_dict().items()}
    lp=OUT/'cache/last_layer.pt'
    if lp.exists():assert state_hash(torch.load(lp,weights_only=True))==state_hash(last)
    else:torch.save(last,lp)
    entries=[]
    for sid in ctx.ids:
        path=OUT/f'cache/subjects/{sid}.pt';texts=[];sources=[]
        for q in range(1,30):
            p=WORK/f'S00/transcripts/B0-37-fulltext/{sid}/{q:02d}.json'
            texts.append(read(p)['answer_text']);sources.append(desc(p))
        if path.exists():
            entry=read(path.with_suffix('.json'));assert entry['sources']==sources;entries.append(entry);continue
        tokens=tokenizer(texts,padding=True,truncation=False,return_tensors='pt').to('cuda')
        assert tokens['input_ids'].shape[1]<=512
        with torch.no_grad():res=model(**tokens,output_hidden_states=True)
        lengths=tokens['attention_mask'].sum(1).tolist();h=res.hidden_states[-2]
        hidden=[h[i,:n].cpu().contiguous().clone() for i,n in enumerate(lengths)]
        full_cls=res.last_hidden_state[:,0].cpu().clone()
        old=np.load(WORK/f'S00/features/B0-37-fulltext/text/{sid}.npy')
        torch.save({'subject_id':sid,'hidden':hidden,'lengths':lengths,'full_cls':full_cls,'fit_ids':[],'labels_used':False},path)
        entry={'subject_id':sid,'cache':desc(path),'sources':sources,'token_lengths':lengths,'fit_ids':[],'labels_used':False,
            'historical_cls_max_abs_difference':float(np.max(np.abs(full_cls.numpy()-old))),'shape':[29,'variable',768]}
        save(path.with_suffix('.json'),entry,immutable=True);entries.append(entry)
        print('CACHE',sid,flush=True)
    save(OUT/'cache_manifest.json',{'created_utc':now(),'subjects':ctx.ids,'rows':entries,'shared_cache_label_free':True,
        'last_layer':desc(lp),'cache_stage':'MacBERT first 11 layers; final layer input','dtype':'float32','attn_implementation':'eager'},immutable=True)

if __name__=='__main__':main()
