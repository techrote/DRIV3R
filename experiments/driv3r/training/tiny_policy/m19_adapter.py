"""Opt-in M19: train SmolLM2-135M attention LoRA, then replay held-out legal game actions.
Requires pinned Hugging Face weights; never runs on ordinary CI/push. Synthetic only.
"""
import argparse, copy, hashlib, json, random, resource, statistics, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from . import m15_pretrained_goal as m15

MODEL = 'smollm2-135m'
MAX_STEPS = 24

class LowRank(nn.Module):
    def __init__(self, base, rank=4):
        super().__init__()
        if not isinstance(base, nn.Linear): raise TypeError('Expected nn.Linear')
        self.base = base.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=5**.5)
        self.scale = 8 / rank
    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale

def inject(model):
    layers = getattr(model, 'layers', None)
    if layers is None: layers = getattr(getattr(model, 'model', None), 'layers', None)
    if layers is None or len(layers)<2: raise RuntimeError('Unknown SmolLM2 layer layout')
    names=[]
    for i in range(len(layers)-2,len(layers)):
        attn=layers[i].self_attn
        for k in ('q_proj','v_proj'):
            setattr(attn,k,LowRank(getattr(attn,k)))
            names.append(f'layers.{i}.self_attn.{k}')
    return names

def adapter_parameters(model):
    return [(n,p) for n,p in model.named_parameters() if p.requires_grad and n.endswith(('.A','.B'))]

def encode(tok, rows):
    return tok([r['text'] for r in rows], return_tensors='pt',
               padding=True, truncation=True, max_length=96)

def hidden(model, tokens):
    t=model(**tokens, use_cache=False).last_hidden_state
    idx=tokens['attention_mask'].sum(-1)-1
    return t[torch.arange(len(idx)),idx].float()

def train(model,tok,initial,rows,batches,lora):
    head=copy.deepcopy(initial)
    params=list(head.parameters()) + ([p for _,p in adapter_parameters(model)] if lora else [])
    optim=torch.optim.AdamW(params,lr=.001 if lora else .006)
    history=[]
    for ids in batches:
        batch=[rows[i] for i in ids]
        tokbatch=encode(tok,batch)
        if lora: x=hidden(model,tokbatch)
        else:
            with torch.no_grad(): x=hidden(model,tokbatch)
        mask=torch.tensor([[a in r['legal'] for a in range(5)] for r in batch])
        y=torch.tensor([r['label'] for r in batch])
        assert mask.gather(1,y[:,None]).all()
        logits=head(x).masked_fill(~mask,-1e6)
        loss=F.cross_entropy(logits,y)
        optim.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params,1.0);optim.step()
        history.append(float(loss.detach()))
    return head.eval().requires_grad_(False),history

def evaluate(model,tok,heads,output,steps):
    m15.BUDGET=24  # bounded held-out action budget; all arms identical
    rows=[]
    # Use the pre-existing M15 action-receipted replay and fixed EVAL topology;
    # normalization is identity, not fitted on evaluation examples.
    dim=model.config.hidden_size
    mu=torch.zeros(dim);sd=torch.ones(dim)
    saved=[(p,p.detach().clone()) for n,p in adapter_parameters(model) if n.endswith('.B')]
    for seed in m15.SEEDS['eval'][:2]:
        for family in m15.FAMILIES:
            for name in ('stock','frozen','lora','oracle','greedy','random'):
                with torch.no_grad():
                    for p,original in saved: p.copy_(original if name=='lora' else torch.zeros_like(original))
                opts={k:(v,mu,sd) for k,v in heads.items()}
                out=output/'rollouts'/f'{seed}-{family}-{name}'
                row=m15.trajectory(torch,MODEL,seed,family,name,opts,tok,model,out)
                rows.append(row)
    return rows

def probe(model,tok,head,rows):
    raw_illegal=0; agreement=0; samples=[]
    for r in rows[:24]:
        t0=time.perf_counter()
        with torch.inference_mode(): score=head(hidden(model,encode(tok,[r]))) [0]
        samples.append((time.perf_counter()-t0)*1000)
        raw_illegal+=int(int(score.argmax()) not in r['legal'])
        legal=max(r['legal'],key=lambda a:(float(score[a]),-a))
        agreement+=int(legal==r['label'])
    return {'n':len(samples),'raw_illegal':raw_illegal,
            'masked_agreement':round(agreement/len(samples),4),
            'p50_ms':round(statistics.median(samples),3),
            'p95_ms':round(sorted(samples)[int(.95*(len(samples)-1))],3)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=MAX_STEPS)
    a=parser.parse_args()
    if not 1<=a.steps<=MAX_STEPS: parser.error('1..24 steps only')
    if a.out.exists(): parser.error('output must not exist')
    a.out.mkdir(parents=True);(a.out/'rollouts').mkdir()
    report={'model':MODEL,'status':'STARTED','synthetic_only':True,'optimizer_updates':0}
    try:
        from transformers import AutoModel,AutoTokenizer
        from huggingface_hub import HfApi
        from safetensors.torch import save_file
        torch.manual_seed(19001);torch.set_num_threads(2)
        ident,rev,license_name=m15.MODELS[MODEL]
        if HfApi().model_info(ident,revision=rev).sha!=rev: raise RuntimeError('revision mismatch')
        tok=AutoTokenizer.from_pretrained(ident,revision=rev,trust_remote_code=False)
        if tok.pad_token is None:tok.pad_token=tok.eos_token
        model=AutoModel.from_pretrained(ident,revision=rev,use_safetensors=True,trust_remote_code=False)
        model.eval().requires_grad_(False)
        names=inject(model)
        trainrows=m15.collect('train')
        report.update({'model_id':ident,'revision':rev,'license':license_name,'trained_names':names,
          'training_id_sha256':hashlib.sha256(json.dumps([r['id'] for r in trainrows]).encode()).hexdigest(),
          'trainable_params':sum(p.numel() for _,p in adapter_parameters(model))})
        initial=nn.Linear(model.config.hidden_size,5)
        stock=copy.deepcopy(initial).eval().requires_grad_(False)
        rng=random.Random(19001)
        batches=[[rng.randrange(len(trainrows)) for _ in range(2)] for _ in range(a.steps)]
        frozen,control_loss=train(model,tok,initial,trainrows,batches,False)
        lora,adapter_loss=train(model,tok,initial,trainrows,batches,True)
        report['optimizer_updates']=2*a.steps
        report['real_adapter_updates']=a.steps
        report['train_loss']={'frozen':control_loss,'lora':adapter_loss}
        dev=m15.collect('dev')
        saved_b=[(p,p.detach().clone()) for n,p in adapter_parameters(model) if n.endswith('.B')]
        with torch.no_grad():
            for p,_ in saved_b:p.zero_()
        report['dev_frozen']=probe(model,tok,frozen,dev)
        with torch.no_grad():
            for p,v in saved_b:p.copy_(v)
        report['dev_lora']=probe(model,tok,lora,dev)
        tensors={n:p.detach().cpu().contiguous() for n,p in adapter_parameters(model)}
        tensors.update({f'head.{n}':p.detach().cpu().contiguous() for n,p in lora.state_dict().items()})
        target=a.out/'m19-adapter.safetensors';save_file(tensors,str(target))
        report['checkpoint_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
        result=evaluate(model,tok,{'stock':stock,'frozen':frozen,'lora':lora},a.out,a.steps)
        (a.out/'rows.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        report['summary']={k:{metric:round(statistics.mean(r[metric] for r in result if r['policy']==k),3)
                              for metric in ('cells','waypoints','branches','retraces')}
                           for k in ('stock','frozen','lora','oracle','greedy','random')}
        report['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['status']='COMPLETED'
    except Exception as e:
        report['status']='FAILED';report['failure']=f'{type(e).__name__}: {e}'
        raise
    finally:
        (a.out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        (a.out/'manifest.json').write_text(json.dumps({str(p.relative_to(a.out)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(a.out.rglob('*')) if p.is_file() and p.name!='manifest.json'},indent=2)+'\n')
if __name__=='__main__':main()
