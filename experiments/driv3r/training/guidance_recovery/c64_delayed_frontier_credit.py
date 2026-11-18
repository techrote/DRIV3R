"""C64: delayed host-observed frontier credit, independent graph transfer.

Synthetic CyberSand-like host only; *not* the native actor/ABI. All features come
from current host-legal candidate actions, visited menus, or published goals.
Neither policy inference nor reward calculation accesses hidden maze adjacency.
Research-only CPU optimizer with a matched wrong-credit control.
"""
from __future__ import annotations
import argparse, base64, collections, copy, hashlib, json, sys, zlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT/'frozen/research-source'))
sys.path.insert(0,str(HERE))
import c60_remote_frontier_rl as c60
import c63_counterfactual_gate as c63

SEED=640118
HORIZON=106
UPDATES=24
GAMMA=.94
ARMS=(('none',0.),('backtrack',.35),('backtrack',1.))
DROPS=(0.,.5)
POLICIES=('C18-A0','C34-RESET','C60-FRONTIER','C60-FROZEN','C64-ALIGNED','C64-SHUFFLED')

def sha(b):return hashlib.sha256(b).hexdigest()

def graph_cohorts():
    # Historical exclusions reconstructed by C63, then add C63 train+test.
    oldtrain,oldtest,oldreceipts,history=c63.history_and_cohorts()
    old={r['graph'] for group in oldreceipts for r in group}
    assert len(old)==len(sum(oldreceipts,[]))
    # C63 earlier tests/families and C60-C62 are outside C64 cohorts.
    # Explicit all-historical graph exclusion, including C60 and C61.
    c60_report=json.loads((ROOT/'publication/C60-summary.json').read_text())
    c60_excl=json.loads((ROOT/'publication/C60-historical-graph-exclusions.json').read_text())
    c61_report=json.loads((ROOT/'results/C61-summary.json').read_text())
    past=set(c60_excl['excluded_graphs'])
    past.update(r['graph'] for group in ('train','online') for r in c60_report['cohorts'][group])
    past.update(r['graph'] for group in ('train_graphs','test_graphs') for r in c61_report[group])
    all_known=past|set(history['C62_reconstructed_hashes'])|old
    worldgen=c60.c59.c58.c55.c54.c52.worlds
    fam=c60.c59.c58.c55.c54.c52.TRAIN_FAMILIES
    ffn=c60.c59.c58.c55.c54.c52.c50.c47.c41.world
    ofn=c60.c59.c58.c55.c54.c52.c50.c47.c42.world
    groups=[];receipts=[]
    for families,fn,seed,reps in [
        (('wilson',),ofn,SEED+1001,2),
        (fam,ffn,SEED+2002,1),
        (('kruskal','wilson'),ofn,SEED+3003,2),
        (fam,ffn,SEED+4004,1)]:
        ws,rs=worldgen(families,fn,seed,reps,exclude=all_known)
        ids={r['graph'] for r in rs}
        assert len(ids)==len(rs) and not ids&all_known
        groups.append(ws);receipts.append(rs);all_known.update(ids)
    tr=groups[0]+groups[1];te=groups[2]+groups[3]
    assert not set(r['graph'] for rs in receipts[:2] for r in rs)&set(r['graph'] for rs in receipts[2:] for r in rs)
    return tr,te,receipts,history

def training_reward(decisions,outcome,mode,rng):
    """Delayed credit from actual executed legal host feedback only.

    The horizon-level completion reward and observed novelty are ONLY trainer
    targets, not controller observations. SHUFFLED corrupts action alignment
    *within the same host-executed trajectory* while preserving reward count.
    """
    n=len(decisions)
    if not n:return np.array([],dtype=np.float32)
    local=np.asarray([e['reward'] for e in decisions],dtype=np.float32)
    if mode=='aligned':
        # A bounded eventual-result signal: observed delayed goal completion,
        # visit novelty, and small per-step cost; no shortest-path oracle.
        reward=np.asarray([.20*max(-1,float(e['reward']))-.025 for e in decisions],np.float32)
        if outcome['goal']:reward[-1]+=9.0
        # Reward delayed after 4 actually executed host decisions: restoration
        # of a future observed novel site, not a prediction of hidden space.
        for t in range(4,n):
            if decisions[t]['reward']>=.5:reward[t]+=.7
    elif mode=='shuffled':
        reward=training_reward(decisions,outcome,'aligned',rng)
        # Wrong credit: same rewards, outcome and episode, reassigned in time.
        reward=reward[rng.permutation(n)]
    else:raise ValueError(mode)
    out=np.empty(n,dtype=np.float32);carry=0.
    for t in range(n-1,-1,-1):
        carry=float(reward[t])+GAMMA*carry;out[t]=carry
    return out

def fit(base,opt,episodes,mode):
    torch.manual_seed(SEED+77)
    model=nn.Linear(c60.NFEAT,1,bias=False)
    nn.init.zeros_(model.weight)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.018,weight_decay=.025)
    rng=np.random.default_rng(SEED+800)
    order=rng.integers(0,len(episodes),size=UPDATES)
    logs=[];train_goals=0;action_count=0
    for step,idx in enumerate(order):
        st,arm=episodes[int(idx)]
        weight=model.weight.detach().numpy().reshape(-1).copy()
        result,dec=c60.execute(base,opt,st,arm,'C60-ALIGNED',weight,0.,collect=True)
        if not dec:continue
        ret=training_reward(dec,result,mode,rng)
        action_count+=len(dec);train_goals+=result['goal']
        x=torch.from_numpy(np.stack([d['features'] for d in dec]))
        frozen=torch.from_numpy(np.stack([d['logits'] for d in dec]))
        chosen=torch.tensor([d['action'] for d in dec],dtype=torch.long)
        # Centered, predeclared on-policy trajectory advantages; avoid large
        # delayed returns making policy deterministic.
        adv=(ret-ret.mean())/max(1.,float(ret.std()))
        logits=frozen+c60.BONUS*arm[1]*torch.clamp(model(x).squeeze(-1),-1,1)
        logp=torch.log_softmax(logits,dim=1)
        prob=torch.softmax(logits,dim=1)
        pgloss=-(logp.gather(1,chosen[:,None]).squeeze(1)*torch.from_numpy(adv)).mean()
        base_logp=torch.log_softmax(frozen,dim=1)
        kl=(prob*(logp-base_logp)).sum(-1).mean()
        entropy=-(prob*logp).sum(-1).mean()
        loss=pgloss+.07*kl-.012*entropy
        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),.75);optimizer.step()
        if step in (0,5,11,17,23):logs.append({'update':step+1,'loss':float(loss.detach()),'kl':float(kl.detach()),'entropy':float(entropy.detach()),'n_steps':len(dec)})
    return model,{'mode':mode,'planned_updates':UPDATES,'performed_updates':len(order),
                  'host_actions':action_count,'training_rollout_goals':train_goals,'trace':logs}

def execute_policy(base,opt,st,arm,drop,pol,weights,trace=False):
    if pol=='C60-FROZEN':return c60.execute(base,opt,st,arm,'C60-ALIGNED',weights['C60-FROZEN'],drop,trace=trace)[0]
    if pol.startswith('C64-'):return c60.execute(base,opt,st,arm,'C60-ALIGNED',weights[pol],drop,trace=trace)[0]
    return c60.execute(base,opt,st,arm,pol,np.zeros(c60.NFEAT,np.float32),drop,trace=trace)[0]

def evaluate(base,opt,episodes,weights):
    records=[];traces=[];seen=collections.Counter()
    for st in episodes:
        for drop in DROPS:
            for arm in ARMS:
                outs={p:execute_policy(base,opt,st,arm,drop,p,weights) for p in POLICIES}
                assert all(v['host_rejections']==0 and v['protected_sha']==st[3]['protected_sha'] for v in outs.values())
                if arm[0]=='none':assert len({(v['action_sha'],v['ink_sha']) for v in outs.values()})==1
                r={'graph':st[6]['graph_sha'],'family':st[8],'case':st[9],
                   'seed':st[7],'target':list(st[5]),'arm':list(arm),'drop':drop,
                   'prefix_sha':st[3]['prefix_sha'],'outcomes':outs}
                records.append(r)
                if arm==ARMS[-1] and drop==0. and seen[st[8]]<2:
                    seen[st[8]]+=1
                    for pol in ('C34-RESET','C60-FRONTIER','C60-FROZEN','C64-ALIGNED','C64-SHUFFLED'):
                        z=execute_policy(base,opt,st,arm,drop,pol,weights,trace=True)
                        assert (z['action_sha'],z['ink_sha'])==(outs[pol]['action_sha'],outs[pol]['ink_sha'])
                        traces.append({'graph':r['graph'],'family':r['family'],'policy':pol,'target':r['target'],
                          'start':z['start_pos'],'goal':z['goal'],'path':z['path'],'action_sha':z['action_sha'],'ink_sha':z['ink_sha']})
    return records,traces

def analysis(records):
    agg={};pers={}
    for fam in sorted(set(r['family'] for r in records)):
      for drop in DROPS:
        rr=[r for r in records if r['family']==fam and r['drop']==drop and r['arm'][0]!='none']
        if not rr:continue
        agg[f'{fam}|{drop}']={'episodes':len(rr),'policies':{p:{
          'goals':sum(r['outcomes'][p]['goal'] for r in rr),
          'new_sites':sum(r['outcomes'][p]['new_sites'] for r in rr),
          'paired_wins':sum(r['outcomes'][p]['goal']>r['outcomes']['C34-RESET']['goal'] for r in rr),
          'paired_losses':sum(r['outcomes'][p]['goal']<r['outcomes']['C34-RESET']['goal'] for r in rr)
          } for p in POLICIES}}
    for pol in POLICIES[2:]:
      diffs=[]
      for r in records:
        if r['arm'][0]=='none' or r['drop']!=0.:continue
        a=r['outcomes'][pol]['personality20'];b=r['outcomes']['C34-RESET']['personality20']
        if a and b:diffs.append({k:a[k]-b[k] for k in ('turn_rate','reverse_rate','unique_ratio','direction_entropy','paint_fraction')})
      pers[pol]={'matched20':len(diffs),'mean_delta':{k:float(np.mean([d[k] for d in diffs])) for k in diffs[0]} if diffs else {}}
    boot={}
    for pol in POLICIES[2:]:
      diffs=collections.defaultdict(int)
      for r in records:
        if r['arm'][0]!='none' and r['drop']==0.:
          diffs[r['graph']]+=r['outcomes'][pol]['goal']-r['outcomes']['C34-RESET']['goal']
      ds=np.asarray(list(diffs.values()),dtype=int)
      rng=np.random.default_rng(SEED+len(pol))
      b=ds[rng.integers(0,len(ds),(2000,len(ds)))].sum(axis=1)
      boot[pol]={'delta_goals':int(ds.sum()),'graph_cluster95_descriptive':np.quantile(b,[.025,.975]).tolist(),
        'positive_graphs':int((ds>0).sum()),'negative_graphs':int((ds<0).sum())}
    return agg,pers,boot

def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    c60.SEED=SEED;c60.HORIZON=HORIZON;c60.c59.c58.c55.TICKS=(9,)
    print('C64 loading prerequisites',flush=True)
    base,opt,_=c60.c59.c58.c55.c54.c52.c50.c47.c44.prerequisites(ROOT/'frozen/predecessor_C42')
    print('C64 generating cohorts',flush=True)
    train,test,receipts,history=graph_cohorts()
    statefn=c60.c59.c58.c56.states_for_goal
    training,trskips=statefn(base,train,'remote');heldout,teskips=statefn(base,test,'remote')
    assert len(training)>=8 and len(heldout)>=8
    trids={s[6]['graph_sha'] for s in training};teids={s[6]['graph_sha'] for s in heldout}
    assert not trids&teids
    ep=[(s,a) for s in training for a in ARMS[1:]]
    print('C64 training heads',len(ep),flush=True)
    aligned,fit_a=fit(base,opt,ep,'aligned')
    shuffled,fit_s=fit(base,opt,ep,'shuffled')
    # Independent exact determinism refits; these are retraining, not only loading.
    aligned2,_=fit(base,opt,ep,'aligned');shuffled2,_=fit(base,opt,ep,'shuffled')
    assert torch.equal(aligned.weight,aligned2.weight) and torch.equal(shuffled.weight,shuffled2.weight)
    prior=json.loads((ROOT/'publication/C60-summary.json').read_text())
    w={'C60-FROZEN':np.asarray(prior['weights']['C60-ALIGNED'],np.float32),
       'C64-ALIGNED':aligned.weight.detach().numpy().flatten().copy(),
       'C64-SHUFFLED':shuffled.weight.detach().numpy().flatten().copy()}
    cp={}
    for pol,mdl in [('C64-ALIGNED',aligned),('C64-SHUFFLED',shuffled)]:
        filename=out/(pol+'.safetensors');save_file(mdl.state_dict(),str(filename));cp[pol]=sha(filename.read_bytes())
    print('C64 evaluating',len(heldout),'heldout states',flush=True)
    records,traces=evaluate(base,opt,heldout,w)
    agg,personality,boot=analysis(records)
    unguided=sum(r['arm'][0]=='none' for r in records)
    sources={p:sha((HERE/p).read_bytes()) for p in ('c64_delayed_frontier_credit.py','c63_counterfactual_gate.py')}
    sources['c60_remote_frontier_rl.py']=sha((ROOT/'frozen/research-source/c60_remote_frontier_rl.py').read_bytes())
    summary={'schema':'driv3r-c64-delayed-frontier-credit/1','seed':SEED,'horizon':HORIZON,
       'sources':sources,'prior_C60_checkpoint_sha256':prior['checkpoint_sha256']['ALIGNED'],
       'checkpoints':cp,'graph_receipts':receipts,'known_C60_C63_exclusions':len(set(json.loads((ROOT/'publication/C60-historical-graph-exclusions.json').read_text())['excluded_graphs'])|set(history['C62_reconstructed_hashes'])|{r['graph'] for rs in c63.history_and_cohorts()[2] for r in rs}),
       'eligible_training':len(training),'eligible_test':len(heldout),'training_skips':trskips,'heldout_skips':teskips,
       'optimizer':{'aligned':fit_a,'shuffled':fit_s},'weights':{k:v.tolist() for k,v in w.items()},
       'matched_condition_rows':len(records),'independently_executed_suffixes':len(records)*len(POLICIES),
       'unguided_parity_conditions':unguided,'replay_traces':len(traces),
       'aggregate':agg,'personality20':personality,'bootstrap':boot,
       'limitations':['Synthetic host only; no native CyberSand acceptance',
         'No graph oracle in features/action selection; long-return rewards computed after execution',
         'Wrong-credit permutation is within trajectories, not full independent randomized training data',
         'Personality proxies are short-horizon and do not establish whole-trajectory preservation',
         'Exploratory graph-bootstrap intervals, not preregistered confirmatory confidence',
         'Checkpoints retained in local owner archive, not immutable Actions artifact storage']}
    ledger={'schema':'driv3r-c64-executed-ledger/1','summary':summary,'records':records,'traces':traces}
    raw=json.dumps(ledger,sort_keys=True,separators=(',',':')).encode()
    (out/'C64-executed-ledger.zlib.b64').write_text(base64.b64encode(zlib.compress(raw,9)).decode()+'\n')
    (out/'C64-summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    (out/'C64-traces.json').write_text(json.dumps(traces,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'training':len(training),'heldout':len(heldout),'suffixes':len(records)*len(POLICIES),
      'unguided':unguided,'traces':len(traces),'bootstrap':boot,
      'aggregate':{k:{p:a['goals'] for p,a in v['policies'].items()} for k,v in agg.items()}},indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();run(a.output)
