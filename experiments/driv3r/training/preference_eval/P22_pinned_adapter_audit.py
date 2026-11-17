"""P22 read-only historical full-LFM350 adapter/greedy-prefix audit.

Inputs: source-verified GP350-012 original Actions ZIPs. No new root SLM
inference, optimisation, human labels, or native CyberSand assertion.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import zipfile

import p21_real_lfm_bestiary as prior

PARENT='436159bd6508063221d31d46f057b5fba7ee5b6f05b27edf003c54e76863898c'
STEP24='047b2bacb61f6f04fb359e4d9839f6c198a17ba7db6dfb451d88e48846c66430'
PARENT_PATH='original-parent/original-experiment/adapter-step-96/adapter_model.safetensors'
CHILD_PATH='original-parent/original-experiment/adapter-step-{step}/adapter_model.safetensors'
NESTED_PATH='original-parent/original-parent/original-experiment/adapter-step-96/adapter_model.safetensors'
ACTIONS=('STRAIGHT','LEFT','RIGHT','BRANCH','RETRACE','STOP')

def digest(b):return hashlib.sha256(b).hexdigest()

def legal_margin(d,action='BRANCH'):
    legal=d['obs']['legal']
    if action not in legal:return None
    z=d['six_logits']
    return z[ACTIONS.index(action)]-max(z[ACTIONS.index(a)] for a in legal if a!=action)

def weight_audit(parent_zip,child_zip):
    from safetensors.torch import load
    with zipfile.ZipFile(parent_zip) as a,zipfile.ZipFile(child_zip) as b:
        baseline=a.read(PARENT_PATH)
        if baseline!=b.read(NESTED_PATH) or digest(baseline)!=PARENT:
            raise ValueError('nested GP006 parent checkpoint custody mismatch')
        snapshots={96:baseline,**{n:b.read(CHILD_PATH.format(step=n)) for n in (24,48,72)}}
    if digest(snapshots[24])!=STEP24:raise ValueError('step24 checkpoint digest mismatch')
    tensors={n:load(buf) for n,buf in snapshots.items()}
    keys=set(tensors[96]); modules=[]
    for layer in (2,5,8,10,12,14):
        for proj in ('q_proj','v_proj'):
            stem=f'base_model.model.model.layers.{layer}.self_attn.{proj}.lora_'
            modules.extend((stem+'A.weight',stem+'B.weight'))
    if set(modules)!=keys or len(keys)!=24 or any(set(v)!=keys for v in tensors.values()):
        raise ValueError('unexpected LoRA targets')
    for name in keys:
        shape=tensors[96][name].shape
        if any(t[name].shape!=shape or not bool(t[name].isfinite().all()) for t in tensors.values()):
            raise ValueError('nonfinite or shape-drift LoRA')
    report=[]
    for layer in (2,5,8,10,12,14):
        for proj in ('q_proj','v_proj'):
            stem=f'base_model.model.model.layers.{layer}.self_attn.{proj}.lora_'
            def effective(n):
                return 2*(tensors[n][stem+'B.weight'].double()@tensors[n][stem+'A.weight'].double())
            initial=effective(96); norm=float(initial.norm()); row={'layer':layer,'projection':proj,'parent_frobenius':round(norm,7)}
            for n in (24,48,72):
                delta=float((effective(n)-initial).norm())
                row[f'step{n}_delta_frobenius']=round(delta,7)
                row[f'step{n}_relative_delta']=round(delta/max(norm,1e-15),7)
            report.append(row)
    return {'sha256_by_checkpoint':{str(n):digest(buf) for n,buf in snapshots.items()},
            'trainable_elements':sum(t.numel() for t in tensors[96].values()),
            'rank':8,'alpha':16,'target_matrices':12,'target_metrics':report}

def prefix_audit(parent,child):
    if set(parent)!=set(child):raise ValueError('unmatched worlds')
    rows=[]
    for world in sorted(parent):
        a,b=parent[world],child[world]
        p,q=a['decisions'],b['decisions']
        if len(p)!=128 or len(q)!=128:raise ValueError('non-128 original episode')
        first=next((i for i in range(128) if p[i]['applied']!=q[i]['applied']),128)
        for i in range(min(first+1,128)):
            if p[i]['obs']!=q[i]['obs']:raise ValueError('counterfactual obs mismatch')
            for d in (p[i],q[i]):
                if d['applied']!=max(d['obs']['legal'],key=lambda name:d['six_logits'][ACTIONS.index(name)]):
                    raise ValueError('archived greedy legal choice mismatch')
        if a['positions'][:first+1]!=b['positions'][:first+1]:raise ValueError('prefix path mismatch')
        valid=[(legal_margin(p[i]),legal_margin(q[i])) for i in range(min(first+1,128))
               if 'BRANCH' in p[i]['obs']['legal']]
        if not valid:raise ValueError('no shared legally comparable branch observations')
        branch=[i for i,d in enumerate(q) if d['applied']=='BRANCH']
        rows.append({'size':world[0],'seed':world[1],'family':world[2],
          'first_different_action_tick':None if first==128 else first,
          'common_legal_branch_logits':len(valid),
          'parent_branch_margin_mean':round(mean(t[0] for t in valid),6),
          'child_branch_margin_mean':round(mean(t[1] for t in valid),6),
          'margin_shift':round(mean(t[1]-t[0] for t in valid),6),
          'parent_branch_wins':sum(t[0]>0 for t in valid),
          'child_branch_wins':sum(t[1]>0 for t in valid),
          'child_branch_ticks':branch,
          'parent_unique_cells':a['metrics']['distinct_cells'],
          'child_unique_cells':b['metrics']['distinct_cells'],
          'same_full_path':a['positions']==b['positions'],
          'parent_json_sha256':a['source_json_sha256'],
          'child_json_sha256':b['source_json_sha256']})
    return rows

def run(archives,out):
    out=Path(out)
    if out.exists():raise ValueError('result directory already exists')
    original={k:prior.load(k,archives[k]) for k in ('stock','gp350006','gp350010')}
    rows=prefix_audit(original['gp350006'],original['gp350010'])
    weights=weight_audit(archives['gp350006'],archives['gp350010'])
    out.mkdir(parents=True)
    for i in range(0,12,2):
        worlds=sorted(original['stock'])[i:i+2];data=[]
        for w in worlds:
            # Only original applied positions, no source identity in the page.
            order=sorted(('gp350006','gp350010'),key=lambda name:digest((str(w)+name+'P22').encode()))
            data.append((w,[(chr(65+j),original[name][w]) for j,name in enumerate(order)]))
        (out/f'P22_BLIND_A4_{i//2+1:02d}.svg').write_text(
            prior.svg_page(data,{w:f'World W{sorted(original["stock"]).index(w)+1:02d}' for w in worlds}))
    result={'schema':'driv3r.preference_eval.p22/1','date_utc':'2026-11-17',
      'original_model':'LiquidAI/LFM2.5-350M','revision':prior.MODEL_REVISION,
      'archive_sha256':{k:digest(Path(v).read_bytes()) for k,v in archives.items()},
      'original_closed_loop_episodes_verified':36,'historical_host_legal_decisions':4608,
      'worlds':12,'independent_seeds':2,'first_divergence_worlds':sum(not r['same_full_path'] for r in rows),
      'shared_legal_branch_logit_pairs':sum(r['common_legal_branch_logits'] for r in rows),
      'positive_branch_decisions_on_shared_states_parent':sum(r['parent_branch_wins'] for r in rows),
      'positive_branch_decisions_on_shared_states_child':sum(r['child_branch_wins'] for r in rows),
      'by_family':{f:{'changed_worlds':sum(r['family']==f and not r['same_full_path'] for r in rows),
          'mean_world_margin_shift':round(mean(r['margin_shift'] for r in rows if r['family']==f),6),
          'total_child_branches':sum(len(r['child_branch_ticks']) for r in rows if r['family']==f)}
          for f in ('architecture','greebles','pipework')},
      'weights':weights,'human_or_owner_ratings':0,'new_pretrained_forwards':0,
      'new_optimizer_steps':0,'new_root_weights_available':False,
      'native_cybersand_gameplay':False,
      'caveat':'These are real historical LoRA checkpoint tensors and original full-model GP350-012 logits, NOT fresh inference. Same-source shared prefix only; after diverging the model observations no longer match. Two independent seeds; no aesthetic-preference generalisation claim.'}
    (out/'P22_RESULTS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (out/'P22_WORLD_DETAILS.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
    manifest={p.name:digest(p.read_bytes()) for p in sorted(out.iterdir())}
    (out/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ('stock','gp350006','gp350010'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    print(json.dumps(run({k:getattr(args,k) for k in ('stock','gp350006','gp350010')},args.out),indent=2))
