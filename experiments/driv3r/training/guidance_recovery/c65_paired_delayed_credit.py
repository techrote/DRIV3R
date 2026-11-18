"""C65: matched-trajectory delayed-credit counterfactual training, synthetic host.

Offline clipped actor updates from IDENTICAL previously executed, host-legal
behaviour trajectories. This is not an on-policy training or native CyberSand
experiment. Evaluations are fresh host rollouts, disjoint by graph identity.
"""
from __future__ import annotations
import argparse,base64,collections,copy,hashlib,json,math,sys,zlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'frozen/research-source'))
sys.path.insert(0,str(HERE))
import c60_remote_frontier_rl as c60
import c63_counterfactual_gate as c63
import c64_delayed_frontier_credit as c64

SEED=650118
UPDATES=48
ARMS=(('none',0.),('backtrack',.35),('backtrack',1.))
DROPS=(0.,.5)
POLICIES=('C18-A0','C34-RESET','C60-FRONTIER','C60-FROZEN','C65-ALIGNED','C65-SHIFT')

def sha(b):return hashlib.sha256(b).hexdigest()

def cohorts():
    oldtrain,oldtest,oldreceipts,hist=c63.history_and_cohorts()
    c60old=json.loads((ROOT/'publication/C60-historical-graph-exclusions.json').read_text())
    c60rep=json.loads((ROOT/'publication/C60-summary.json').read_text())
    c61rep=json.loads((ROOT/'results/C61-summary.json').read_text())
    c64rep=json.loads((ROOT/'lab/C64-summary.json').read_text())
    prior=set(c60old['excluded_graphs'])
    prior.update(r['graph'] for mode in ('train','online') for r in c60rep['cohorts'][mode])
    prior.update(r['graph'] for mode in ('train_graphs','test_graphs') for r in c61rep[mode])
    prior.update(r['graph'] for group in oldreceipts for r in group)
    prior.update(hist['C62_reconstructed_hashes'])
    prior.update(r['graph'] for group in c64rep['graph_receipts'] for r in group)
    original=len(prior)
    worldgen=c60.c59.c58.c55.c54.c52.worlds
    familiar=c60.c59.c58.c55.c54.c52.TRAIN_FAMILIES
    familiarfn=c60.c59.c58.c55.c54.c52.c50.c47.c41.world
    unseenfn=c60.c59.c58.c55.c54.c52.c50.c47.c42.world
    groups=[];receipts=[]
    for families,worldfn,seed,reps in [
        (('wilson',),unseenfn,SEED+101,3),
        (familiar,familiarfn,SEED+202,2),
        (('wilson','kruskal'),unseenfn,SEED+303,3),
        (familiar,familiarfn,SEED+404,2)]:
        w,r=worldgen(families,worldfn,seed,reps,exclude=prior)
        ids=[x['graph'] for x in r]
        assert len(ids)==len(set(ids)) and not set(ids)&prior
        groups.append(w);receipts.append(r);prior.update(ids)
    train=groups[0]+groups[1];test=groups[2]+groups[3]
    assert not {r['graph'] for g in receipts[:2] for r in g}&{r['graph'] for g in receipts[2:] for r in g}
    return train,test,receipts,original

def returns_for(dec,outcome,mode):
    # Reward signal depends only on later host-executed actions/goal success.
    # Negative control circularly shifts the EXACT SAME reward vector by >=half
    # the episode; per-episode count, goal outcome and return distribution fixed.
    n=len(dec)
    reward=np.asarray([.20*float(x['reward'])-.025 for x in dec],np.float32)
    if outcome['goal']:reward[-1]+=9.
    if mode=='shift' and n>1:reward=np.roll(reward,max(1,n//2))
    elif mode!='aligned' and mode!='shift':raise ValueError(mode)
    ret=np.empty(n,np.float32);carry=0.
    for i in range(n-1,-1,-1):
        carry=float(reward[i])+.94*carry;ret[i]=carry
    return ret

def collect(base,opt,states,prior):
    # Every arm shares the same executed host episode for BOTH training heads.
    # Behaviour policy is the frozen C60 neural policy. No optimizer rollouts.
    episodes=[];summary=[]
    for st in states:
        for arm in ARMS[1:]:
            outcome,dec=c60.execute(base,opt,st,arm,'C60-ALIGNED',prior,0.,collect=True)
            if not dec:continue
            aligned=returns_for(dec,outcome,'aligned')
            shifted=returns_for(dec,outcome,'shift')
            assert len(dec)==len(aligned)==len(shifted)
            episodes.append((dec,aligned,shifted))
            summary.append({'graph':st[6]['graph_sha'],'family':st[8],'arm':arm[1],
                'steps':len(dec),'goal':outcome['goal'],'action_sha':outcome['action_sha']})
    return episodes,summary

def flatten(episodes,prior):
    # Ragged host action menus are padded, never filled with illegal candidates.
    maxmenu=max(len(d['features']) for ds,*_ in episodes for d in ds)
    rows=sum(len(ds) for ds,*_ in episodes)
    feat=np.zeros((rows,maxmenu,c60.NFEAT),np.float32)
    logits=np.full((rows,maxmenu),-1e7,np.float32)
    acts=np.empty(rows,np.int64);strength=np.empty(rows,np.float32)
    aligned=np.empty(rows,np.float32);shifted=np.empty(rows,np.float32)
    idx=0
    for dec,a,b in episodes:
        for j,d in enumerate(dec):
            n=len(d['features'])
            feat[idx,:n]=d['features'];logits[idx,:n]=d['logits']
            acts[idx]=d['action'];strength[idx]=d['strength']
            aligned[idx]=a[j];shifted[idx]=b[j];idx+=1
    assert idx==rows
    frozen=torch.from_numpy(logits)
    x=torch.from_numpy(feat)
    s=torch.from_numpy(strength)
    b=torch.from_numpy(np.asarray(prior,np.float32))
    with torch.no_grad():
        behavior_log=torch.log_softmax(frozen+c60.BONUS*s[:,None]*torch.clamp(x@b,-1,1),-1)
    return {'feat':x,'logits':frozen,'acts':torch.from_numpy(acts),'strength':s,
            'behavior_log':behavior_log,'aligned':aligned,'shift':shifted,
            'menus':maxmenu,'rows':rows}

def train(data,prior,mode):
    # Identical exposure, minibatch order and update budget. Only return credit
    # assignment differs. Clipped off-policy importance weights, bounded bias.
    torch.manual_seed(SEED+777)
    mdl=nn.Linear(c60.NFEAT,1,bias=False)
    with torch.no_grad():mdl.weight.copy_(torch.from_numpy(np.asarray(prior,np.float32))[None,:])
    optim=torch.optim.AdamW(mdl.parameters(),lr=.008,weight_decay=.012)
    n=data['rows'];rng=np.random.default_rng(SEED+30303)
    raw=data[mode];adv=(raw-raw.mean())/max(1.,float(raw.std()))
    at=torch.from_numpy(adv)
    logs=[]
    for step in range(UPDATES):
        ix=torch.from_numpy(rng.integers(0,n,size=min(n,192),dtype=np.int64))
        x=data['feat'][ix];f=data['logits'][ix];s=data['strength'][ix]
        new=f+c60.BONUS*s[:,None]*torch.clamp(mdl(x).squeeze(-1),-1.,1.)
        lp=torch.log_softmax(new,-1)
        a=data['acts'][ix,None]
        diff=lp.gather(1,a).squeeze(1)-data['behavior_log'][ix].gather(1,a).squeeze(1)
        ratio=torch.exp(torch.clamp(diff,-10,10))
        surrogate=torch.minimum(ratio*at[ix],torch.clamp(ratio,.8,1.2)*at[ix])
        p=torch.softmax(new,-1)
        kl=(p*(lp-data['behavior_log'][ix])).sum(-1).mean()
        entropy=-(p*lp).sum(-1).mean()
        loss=-surrogate.mean()+.05*kl-.007*entropy
        optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(mdl.parameters(),.5);optim.step()
        if step in (0,11,23,35,47):
            logs.append({'step':step+1,'loss':float(loss.detach()),'kl':float(kl.detach()),
                'clipped_fraction':float(((ratio<.8)|(ratio>1.2)).float().mean())})
    return mdl,{'updates':UPDATES,'shared_training_actions':n,'mean_aligned_return':float(data['aligned'].mean()),
        'matched_minibatches':UPDATES,'trace':logs}

def policy(base,opt,st,arm,drop,name,weights,trace=False):
    if name=='C60-FROZEN':return c60.execute(base,opt,st,arm,'C60-ALIGNED',weights[name],drop,trace=trace)[0]
    if name.startswith('C65-'):return c60.execute(base,opt,st,arm,'C60-ALIGNED',weights[name],drop,trace=trace)[0]
    return c60.execute(base,opt,st,arm,name,np.zeros(c60.NFEAT,np.float32),drop,trace=trace)[0]

def path_metrics(z):
    ps=[tuple(z['start_pos'])]+[tuple(t['p']) for t in z['path']]
    steps=[(b[0]-a[0],b[1]-a[1]) for a,b in zip(ps,ps[1:])]
    travel=[x for x in steps if x!=(0,0)]
    turns=sum(a!=b for a,b in zip(travel,travel[1:]))
    reversal=sum(a[0]==-b[0] and a[1]==-b[1] for a,b in zip(travel,travel[1:]))
    counts=collections.Counter(travel);n=max(1,len(travel))
    entropy=-sum((v/n)*math.log2(v/n) for v in counts.values()) if counts else 0.
    return {'steps':len(steps),'turn_rate':turns/max(1,len(travel)-1),
      'reverse_rate':reversal/max(1,len(travel)-1),'direction_entropy':entropy,
      'unique_ratio':len(set(ps))/max(1,len(ps)),
      'paint_fraction':sum(t['a']=='INK' for t in z['path'])/max(1,len(steps))}

def evaluate(base,opt,states,weights):
    records=[];sample=[];seen=collections.Counter()
    for si,st in enumerate(states):
        if si%2==0:print('C65 evaluation state',si,'/',len(states),flush=True)
        for drop in DROPS:
            for arm in ARMS:
                out={p:policy(base,opt,st,arm,drop,p,weights,trace=True) for p in POLICIES}
                assert all(v['host_rejections']==0 and v['protected_sha']==st[3]['protected_sha'] for v in out.values())
                if arm[0]=='none':assert len({(v['action_sha'],v['ink_sha']) for v in out.values()})==1
                record={'graph':st[6]['graph_sha'],'family':st[8],'case':st[9],
                  'seed':st[7],'arm':list(arm),'drop':drop,'prefix_sha':st[3]['prefix_sha'],
                  'outcomes':{p:{k:v for k,v in z.items() if k!='path'}|{'morphology':path_metrics(z)} for p,z in out.items()}}
                records.append(record)
                if drop==0. and arm==ARMS[-1] and seen[st[8]]<1:
                    seen[st[8]]+=1
                    for p in POLICIES:
                        z=policy(base,opt,st,arm,drop,p,weights,trace=True)
                        assert z['action_sha']==out[p]['action_sha'] and z['ink_sha']==out[p]['ink_sha']
                        sample.append({'graph':record['graph'],'family':st[8],'policy':p,'goal':z['goal'],
                          'target':list(st[5]),'start':z['start_pos'],'action_sha':z['action_sha'],
                          'ink_sha':z['ink_sha'],'path':z['path']})
    return records,sample

def analyse(rows):
    agg={}
    for drop in DROPS:
        rr=[r for r in rows if r['drop']==drop and r['arm'][0]!='none']
        agg[str(drop)]={'episodes_per_policy':len(rr),'policies':{p:{
          'goals':sum(r['outcomes'][p]['goal'] for r in rr),
          'new_sites':sum(r['outcomes'][p]['new_sites'] for r in rr),
          'wins_vs_reset':sum(r['outcomes'][p]['goal']>r['outcomes']['C34-RESET']['goal'] for r in rr),
          'losses_vs_reset':sum(r['outcomes'][p]['goal']<r['outcomes']['C34-RESET']['goal'] for r in rr),
          'family_goals':{fam:sum(r['outcomes'][p]['goal'] for r in rr if r['family']==fam)
            for fam in sorted(set(x['family'] for x in rr))}} for p in POLICIES}}
    morphology={}
    for p in ('C60-FRONTIER','C60-FROZEN','C65-ALIGNED','C65-SHIFT'):
        rr=[r for r in rows if r['drop']==0. and r['arm'][0]!='none']
        metrics=('turn_rate','reverse_rate','direction_entropy','unique_ratio','paint_fraction')
        morphology[p]={'n':len(rr),'mean_delta_vs_reset':{k:float(np.mean([
          r['outcomes'][p]['morphology'][k]-r['outcomes']['C34-RESET']['morphology'][k] for r in rr])) for k in metrics}}
    rng=np.random.default_rng(SEED+313)
    graphmap=collections.defaultdict(int)
    for r in rows:
        if r['drop']==0. and r['arm'][0]!='none':
            graphmap[r['graph']]+=r['outcomes']['C65-ALIGNED']['goal']-r['outcomes']['C65-SHIFT']['goal']
    delta=np.asarray(list(graphmap.values()),np.int32)
    ci=np.quantile(delta[rng.integers(0,len(delta),size=(5000,len(delta)))].sum(1),[.025,.975]).tolist()
    return agg,morphology,{'aligned_minus_shift':int(delta.sum()),'graph_cluster95_descriptive':ci,
                             'negative_graphs':int((delta<0).sum()),'positive_graphs':int((delta>0).sum())}

def gallery(traces,output):
    pick=traces[:min(30,len(traces))]
    cols=5;cellw=242;cellh=186;rows=math.ceil(len(pick)/cols)
    txt=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{cols*cellw}" height="{rows*cellh}" viewBox="0 0 {cols*cellw} {rows*cellh}">',
         '<rect width="100%" height="100%" fill="#101923"/>']
    for i,z in enumerate(pick):
        ox=(i%cols)*cellw;oy=(i//cols)*cellh
        points=[z['start']]+[q['p'] for q in z['path']]
        xs=[p[0] for p in points]+[z['target'][0]];ys=[p[1] for p in points]+[z['target'][1]]
        lo_x,hi_x=min(xs)-1,max(xs)+1;lo_y,hi_y=min(ys)-1,max(ys)+1
        scale=min(218/max(1,hi_x-lo_x),132/max(1,hi_y-lo_y))
        def pt(p):return f'{ox+10+(p[0]-lo_x)*scale:.1f},{oy+32+(p[1]-lo_y)*scale:.1f}'
        color='#7bd3f7' if z['goal'] else '#e6a66c'
        txt.extend([f'<rect x="{ox+2}" y="{oy+2}" width="238" height="182" rx="6" fill="#192737"/>',
            f'<text x="{ox+9}" y="{oy+20}" fill="#fafafa" font-family="monospace" font-size="11">{z["family"]} {z["policy"]}</text>',
            f'<polyline points="{" ".join(pt(p) for p in points)}" fill="none" stroke="{color}" stroke-width="1.5"/>',
            f'<circle cx="{pt(z["target"]).split(",")[0]}" cy="{pt(z["target"]).split(",")[1]}" r="4" fill="#f2e165"/>',
            f'<text x="{ox+9}" y="{oy+173}" fill="#b9cbd5" font-family="monospace" font-size="10">{len(z["path"])} actions | goal={z["goal"]}</text>'])
    txt.append('</svg>');output.write_text('\n'.join(txt)+'\n')

def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    c60.SEED=SEED;c60.HORIZON=95;c60.c59.c58.c55.TICKS=(9,)
    base,opt,_=c60.c59.c58.c55.c54.c52.c50.c47.c44.prerequisites(ROOT/'frozen/predecessor_C42')
    train,test,receipts,nprior=cohorts()
    states=c60.c59.c58.c56.states_for_goal
    training,train_skips=states(base,train,'remote');testset,test_skips=states(base,test,'remote')
    assert len(training)>=9 and len(testset)>=12
    # Fixed-order family-balanced held-out subset; never select based on outcomes.
    fams=sorted(set(s[8] for s in testset));selection=[]
    for fam in fams:
        selection.extend([s for s in testset if s[8]==fam][:2])
    testset=selection
    print('C65 states:',len(training),len(testset),{f:sum(s[8]==f for s in testset) for f in fams},flush=True)
    p=json.loads((ROOT/'publication/C60-summary.json').read_text())
    frozen=np.asarray(p['weights']['C60-ALIGNED'],np.float32)
    print('C65 collect',flush=True)
    episodes,training_receipts=collect(base,opt,training,frozen)
    assert sum(x['steps'] for x in training_receipts)>500
    print('C65 training actions:',sum(x['steps'] for x in training_receipts),flush=True)
    data=flatten(episodes,frozen)
    aligned,al=train_model('aligned',data,frozen)
    shifted,sh=train_model('shift',data,frozen)
    aligned2,_=train_model('aligned',data,frozen)
    shifted2,_=train_model('shift',data,frozen)
    assert torch.equal(aligned.weight,aligned2.weight) and torch.equal(shifted.weight,shifted2.weight)
    weights={'C60-FROZEN':frozen,'C65-ALIGNED':aligned.weight.detach().numpy().flatten(),
        'C65-SHIFT':shifted.weight.detach().numpy().flatten()}
    checks={}
    for name,mdl in [('C65-ALIGNED',aligned),('C65-SHIFT',shifted)]:
        file=out/(name+'.safetensors');save_file(mdl.state_dict(),str(file));checks[name]=sha(file.read_bytes())
    print('C65 evaluating',len(testset),flush=True)
    records,traces=evaluate(base,opt,testset,weights)
    aggregate,morph,bootstrap=analyse(records)
    sources={name:sha((HERE/name).read_bytes()) for name in ['c65_paired_delayed_credit.py','c64_delayed_frontier_credit.py']}
    sources['c60_remote_frontier_rl.py']=sha((ROOT/'frozen/research-source/c60_remote_frontier_rl.py').read_bytes())
    summary={'schema':'driv3r-c65-paired-delayed-credit/1','seed':SEED,'source_sha256':sources,
       'prior_C60_checkpoint_sha256':p['checkpoint_sha256']['ALIGNED'],
       'checkpoint_sha256':checks,'exclusion_count_before_new':nprior,
       'graph_receipts':receipts,'training_states':len(training),'test_states':len(testset),
       'training_skips':train_skips,'test_skips':test_skips,'training_receipts':training_receipts,
       'behavior_policy':'frozen C60 neural; exact same host-executed trajectories for both heads',
       'training_actions_each':data['rows'],'training_episodes_each':len(episodes),
       'optimizer':{'ALIGNED':al,'SHIFT':sh},'matched_condition_rows':len(records),
       'host_executed_suffixes':len(records)*len(POLICIES),
       'unguided_parity_conditions':sum(r['arm'][0]=='none' for r in records),
       'replayed_traces':len(traces),'aggregate':aggregate,'morphology':morph,'bootstrap':bootstrap,
       'limitations':['Offline clipped importance-sampled updates: not on-policy policy gradient',
          'Synthetic CyberSand-like host, not native CyberSand ABI',
          'Shifted-control retains per-episode success and length but scrambles temporal action credit',
          'No preregistered confirmatory test, graph-cluster bootstrap descriptive only',
          'Full-trajectory morphology is a proxy, not an independent qualitative personality test',
          'Model checkpoint binaries only in local owner archive; immutable Actions custody not established']}
    ledger={'schema':'driv3r-c65-executed-ledger/1','summary':summary,'records':records,'traces':traces}
    (out/'C65-summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    (out/'C65-executed-ledger.zlib.b64').write_text(base64.b64encode(zlib.compress(json.dumps(ledger,sort_keys=True,separators=(',',':')).encode(),9)).decode()+'\n')
    gallery(traces,out/'C65-gallery.svg')
    print(json.dumps({'train_states':len(training),'test_states':len(testset),
       'train_actions':data['rows'],'suffixes':len(records)*len(POLICIES),
       'replayed':len(traces),'aggregate':aggregate,'bootstrap':bootstrap},indent=2),flush=True)
    return summary

# Kept as a distinct function to make identical update budget explicit.
def train_model(mode,data,prior):return train(data,prior,mode)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output');args=ap.parse_args();run(args.output)
