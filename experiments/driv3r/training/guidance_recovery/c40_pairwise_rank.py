"""C40 synthetic, host-legal candidate ranking: matched pairwise-label controls.

Research only; no CyberSand changes, no native ABI claim. Training oracle executes
candidate futures offline. Inference uses only current host-visible candidates.
"""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from c39_action_value import ActionHead, action_features, heldout, evaluate, summarize
from c34_independent_maze import make_world, executed_suffix
from c28_temporal_option import OptionHead
from c29_probe_recovery import PREREQS
from c36_personality_budget import load_c12, a0
from experiment import sha

SEED=400117
COHORTS=(('A', 400101), ('B', 540011), ('C', 680021), ('D', 820031))

def collect(seed=SEED, per_condition=14, steps=3):
    rng=np.random.default_rng(seed+41)
    states=[];receipts=[]
    for case in ('detour','forward'):
      for personality in (-1,1):
       for i in range(per_condition):
        world,info=make_world(seed+17117*i+personality*107+(case=='forward')*991,'tree',case,personality)
        for t in range(steps):
            menu=world.candidates(.35)
            legal=[j for j in range(9) if menu[j].legal]
            if len(legal)<2:break
            returns=np.asarray([executed_suffix(world,j,steps=10) for j in legal],np.float32)
            states.append((action_features(world,menu)[legal],returns))
            receipts.append((info['graph_sha'],sha([world.pos,world.history,world.goal]),sha(returns.tolist())))
            options=[j for j in legal if j<8 and menu[j].target not in world.history]
            j=max(options,key=lambda x:world.distance()-world.distance(menu[x].target)) if options else 8
            assert menu[j].legal and world.apply(menu[j]) and world.invalid_attempts==0
    assert len(states)>100 and len(set(x[0] for x in receipts))==4*per_condition
    return states,receipts

def train(states,seed=SEED,epochs=7):
    torch.set_num_threads(1)
    torch.manual_seed(seed+777)
    init=ActionHead().eval()
    models=[copy.deepcopy(init),copy.deepcopy(init),init]
    rng=np.random.default_rng(seed+41)
    orders=[rng.permutation(len(states)).tolist() for _ in range(epochs)]
    shuffled=[rng.permutation(len(y)) for _,y in states]
    stats={}
    for name,head,permuted in zip(('C40-PAIR','C40-SHUFFLED'),models,(False,True)):
        opt=torch.optim.AdamW(head.parameters(),lr=.005,weight_decay=.01)
        curves=[];updates=0
        for order in orders:
            loss_total=0.
            for i in order:
                features,ret=states[i]
                labels=ret[shuffled[i]] if permuted else ret
                target=torch.softmax(torch.from_numpy(labels)/.35,dim=0)
                score=head(torch.from_numpy(features))
                loss=-(target*torch.log_softmax(score,dim=0)).sum()
                opt.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(),1.)
                opt.step();loss_total+=float(loss.detach());updates+=1
            curves.append(round(loss_total/len(order),7))
        head.eval();stats[name]={'optimizer_updates':updates,'mean_loss_by_epoch':curves}
    return models,stats

def run(root,out,replicate=True):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    states,receipts=collect()
    models,training=train(states)
    if replicate:
        repeat,stats2=train(states)
        assert training==stats2
        assert all(all(torch.equal(m.state_dict()[k],r.state_dict()[k]) for k in m.state_dict()) for m,r in zip(models,repeat))
    else:repeat=[]
    checkpoints={}
    for name,head in zip(('C40-PAIR','C40-SHUFFLED','C40-RANDOM'),models):
        fn=out/(name+'.safetensors')
        save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},str(fn))
        checkpoints[name]=hashlib.sha256(fn.read_bytes()).hexdigest()
    blind,offline=heldout(models,seed=994041,graphs_per_family=14)
    (out/'C40-offline.json').write_text(json.dumps(blind,sort_keys=True)+'\n')
    offline['models']={dict(zip(('C39-ALIGNED','C39-SHUFFLED','C39-RANDOM'),('C40-PAIR','C40-SHUFFLED','C40-RANDOM'))).get(k,k):v for k,v in offline['models'].items()}
    previous=[]
    for label in ('C39-ALIGNED','C39-SHUFFLED','C39-RANDOM'):
        h=ActionHead();h.load_state_dict(load_file(str(root/'reference'/f'{label}.safetensors')));h.eval();previous.append(h)
    _,offline_prev=heldout(previous,seed=994041,graphs_per_family=14)
    base=a0(load_c12([root/'prerequisites'/f'guidance_recovery_{p}.safetensors' for p in PREREQS]))
    option=OptionHead();option.load_state_dict(load_file(str(root/'results/C28-TEMPORAL.safetensors')));option.eval()
    cohorts={};all_rows=[]
    for name,seed in COHORTS:
        path=out/f'C40-cohort-{name}.json'
        if path.exists(): rows=json.loads(path.read_text())
        else:
            rows=evaluate(base,option,models,seed,episodes_per_case=1,horizon=78)
            path.write_text(json.dumps(rows,sort_keys=True)+'\n')
        raw=summarize(rows)
        renamed={'C39-ALIGNED':'C40-PAIR','C39-SHUFFLED':'C40-SHUFFLED','C39-RANDOM':'C40-RANDOM'}
        cohorts[name]={renamed.get(k,k):v for k,v in raw.items()}
        all_rows+=rows
    models_order=('C18-A0','C34-RESET','C40-PAIR','C40-SHUFFLED','C40-RANDOM')
    assert len(all_rows)==len(COHORTS)*240
    assert len({r['graph_sha'] for r in all_rows})==len(COHORTS)*16
    summary={'schema':'driv3r-c40-pairwise/1',
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'training':{'states':len(states),'graphs':len(set(x[0] for x in receipts)),
                  'candidate_examples':sum(len(y) for _,y in states),
                  'receipts_sha256':sha(receipts),'models':training,
                  'replicated_byte_identical':replicate},
      'checkpoint_sha256':checkpoints,'offline':offline,'offline_previous_C39':offline_prev,
      'cohorts':cohorts,'online_episodes':len(all_rows),
      'guided_goal_events':{m:sum(cohorts[c][m]['goal_events'] for c,_ in COHORTS) for m in models_order},
      'condition_regressions_vs_RESET':{m:sum(len(cohorts[c][m]['regressions_vs_C34-RESET']) for c,_ in COHORTS) for m in models_order},
      'no_guidance_exact':{},
      'host_rejections':sum(r['host_rejections'] for r in all_rows),
      'protected_changes':'not measured independently; synthetic host enforces protected set'
    }
    summary['no_guidance_exact']={m:[sum(cohorts[c][m]['unguided_exact'] for c,_ in COHORTS),sum(cohorts[c][m]['unguided_n'] for c,_ in COHORTS)] for m in models_order}
    assert summary['host_rejections']==0
    for m in models_order: assert summary['no_guidance_exact'][m][0]==summary['no_guidance_exact'][m][1]
    (out/'C40-summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('out');a=p.parse_args()
    summary=run(a.root,a.out)
    print(json.dumps({'training':summary['training'],'offline':summary['offline'],
                      'old_C39':summary['offline_previous_C39'],
                      'goals':summary['guided_goal_events'],
                      'regressions':summary['condition_regressions_vs_RESET']},indent=2))
