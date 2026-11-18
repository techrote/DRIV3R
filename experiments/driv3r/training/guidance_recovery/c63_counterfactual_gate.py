"""C63: bounded stochastic episode-level counterfactual recovery gate.

Synthetic CyberSand-like host only. Inference observes executed host history,
current legal menu-derived frontiers, announced goal and host cue. It never
sees hidden edges, protected geometry, training counterfactuals or oracle paths.
Train an ACTUAL small PyTorch model on two host-executed policy outcomes per
training condition; then compare never-seen Wilson/Kruskal and familiar graphs.
"""
from __future__ import annotations
import base64, collections, copy, hashlib, json, math, sys, zlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'frozen/research-source'))
import c60_remote_frontier_rl as c60

SEED=630118
HORIZON=106
TRAIN_REPS_W=5
TRAIN_REPS_F=3
TEST_REPS_OOD=8
TEST_REPS_FAMILIAR=2
ARMS=(('none',0.),('backtrack',.35),('backtrack',1.))
DROPS=(0.,.5)
POLICIES=('C18-A0','C34-RESET','C60-FRONTIER','C60-FROZEN',
          'C63-ALIGNED','C63-SHUFFLED','C63-COIN')
FEATURES=12
UPDATES=128
BATCH=64


def sha256_bytes(b):return hashlib.sha256(b).hexdigest()


def source_hashes():
    names=('c60_remote_frontier_rl.py','c59_eligibility_personality.py',
      'c58_dual_memory_recovery.py','c56_memory_recovery.py','c34_independent_maze.py',
      'c52_shortest_step_cue.py','experiment.py')
    return {n:sha256_bytes((ROOT/'frozen/research-source'/n).read_bytes()) for n in names}


def history_and_cohorts():
    previous=json.loads((ROOT/'publication/C60-summary.json').read_text())
    exclusion=json.loads((ROOT/'publication/C60-historical-graph-exclusions.json').read_text())
    predecessor=json.loads((ROOT/'results/C61-summary.json').read_text())
    past=set(exclusion['excluded_graphs'])
    past.update(r['graph'] for k in ('train','online') for r in previous['cohorts'][k])
    past.update(r['graph'] for k in ('train_graphs','test_graphs') for r in predecessor[k])
    # Reconstruct C62 generated identities from its published fixed seeds and
    # world-construction contract, without using C62 model or outcome results.
    worldgen=c60.c59.c58.c55.c54.c52.worlds
    familiar=c60.c59.c58.c55.c54.c52.TRAIN_FAMILIES
    ffn=c60.c59.c58.c55.c54.c52.c50.c47.c41.world
    ofn=c60.c59.c58.c55.c54.c52.c50.c47.c42.world
    previous_used=set(past)
    c62=[]
    for fs,fn,seed,reps in [
       (('wilson',),ofn,621118,5),
       (familiar,ffn,622118,2),
       (('kruskal','wilson'),ofn,623118,2),
       (('tree','braid'),ffn,624118,1)]:
        _,receipts=worldgen(fs,fn,seed,reps,exclude=previous_used)
        ids=[r['graph'] for r in receipts]
        assert not set(ids)&previous_used
        previous_used.update(ids);c62.extend(ids)
    assert len(c62)==76 and len(set(c62))==len(c62)
    # Publicly published C62 boundary identities; fail rather than silently
    # removing the wrong graphs if the frozen generator changes.
    assert c62[0]=='dd31a132a8f527419aa51ed332318f08acab83762ba69b1791dcc1157007a551'
    assert c62[-1]=='7ad27284af7899379e6157739fd0f8a9c07d6d65378bc71482fe0ccd2c487d82'
    past.update(c62)
    groups=[];rs=[];used=set(past)
    for fams,fn,seed,reps in [
       (('wilson',),ofn,SEED+1001,TRAIN_REPS_W),
       (familiar,ffn,SEED+2002,TRAIN_REPS_F),
       (('wilson','kruskal'),ofn,SEED+3003,TEST_REPS_OOD),
       (familiar,ffn,SEED+4004,TEST_REPS_FAMILIAR)]:
        worlds,receipts=worldgen(fams,fn,seed,reps,exclude=used)
        ids={r['graph'] for r in receipts}
        assert len(ids)==len(receipts) and not ids&used
        used.update(ids);groups.append(worlds);rs.append(receipts)
    train=groups[0]+groups[1];test=groups[2]+groups[3]
    assert not ({r['graph'] for r in rs[0]+rs[1]} & {r['graph'] for r in rs[2]+rs[3]})
    return train,test,rs,{'C60_C61_C62_excluded':len(past),'C62_reconstructed_hashes':c62}


def observe_features(st,arm,drop):
    snap,_,log,_,tick,goal,info,_,_,_,_=st
    w=copy.deepcopy(snap);w.goal=tuple(goal);w.goal_change=True;w.goals_seen+=1;w.stuck=2
    led=c60.c59.c58.damaged(log,info['graph_sha'],tick,drop)
    # Current host observation is allowed, but never consult unpublished future
    # menus, world.protected, world.allowed or map adjacency.
    led.see(w)
    here=tuple(w.pos);front=led.frontiers()
    routes=[led.route(here,p) for p in front]
    lengths=[len(p)-1 for p in routes if p is not None and len(p)>1]
    goal_distance=abs(here[0]-goal[0])+abs(here[1]-goal[1])
    v=np.asarray([1., min(1.,goal_distance/28.),
      min(1.,len(led.visits)/30.),
      min(1.,led.visits[here]/7.),
      min(1.,len(front)/14.),
      float(bool(lengths)),
      min(1.,min(lengths)/12.) if lengths else 1.,
      min(1.,len(led.actions)/20.),
      float(w.personality>0),
      float(arm[1]), float(drop),
      min(1.,len(led.neighbors)/30.)],dtype=np.float32)
    assert len(v)==FEATURES and np.isfinite(v).all()
    return v


def benefit(a,b):
    # Counterfactual delayed outcome only in TRAINER: (FRONTIER - RESET).
    # Goal completion dominates modest exploratory progress and extra steps.
    diff=12.*(a['goal']-b['goal'])+.045*(a['new_sites']-b['new_sites'])
    diff-=.012*(a['steps']-b['steps'])
    diff+=.02*(b['distance']-a['distance'])
    return float(np.clip(diff,-12,12))


def pair_training(base,opt,train):
    rows=[];receipts=[]
    for st in train:
      for drop in DROPS:
        for arm in ARMS[1:]:
          a,_=c60.execute(base,opt,st,arm,'C60-FRONTIER',np.zeros(c60.NFEAT,np.float32),drop)
          b,_=c60.execute(base,opt,st,arm,'C34-RESET',np.zeros(c60.NFEAT,np.float32),drop)
          assert a['host_rejections']==b['host_rejections']==0
          assert a['protected_sha']==b['protected_sha']==st[3]['protected_sha']
          adv=benefit(a,b)
          rows.append({'x':observe_features(st,arm,drop),'adv':adv,'family':st[8],
            'graph':st[6]['graph_sha'],'arm':arm[1],'drop':drop})
          receipts.append({'graph':st[6]['graph_sha'],'family':st[8],'cue':arm[1],
              'drop':drop,'adv':adv,'frontier_goal':a['goal'],'reset_goal':b['goal'],
              'frontier_new':a['new_sites'],'reset_new':b['new_sites'],
              'frontier_action_sha':a['action_sha'],'reset_action_sha':b['action_sha'],
              'protected_sha':a['protected_sha']})
    assert len(rows)>70
    return rows,receipts


def fit(rows,mode):
    # Equal data, equal 128 AdamW steps, equal minibatches & initial weights.
    # Wrong-credit control permutes delayed outcomes *within* family/cue/drop
    # strata; no heldout evaluation outcomes enter training.
    torch.manual_seed(SEED+7)
    head=nn.Linear(FEATURES,1,bias=False)
    nn.init.zeros_(head.weight)
    with torch.no_grad():head.weight[0,0]=-1.0  # conservative abstention prior
    opt=torch.optim.AdamW(head.parameters(),lr=.019,weight_decay=.018)
    xs=np.stack([r['x'] for r in rows]).astype(np.float32)
    adv=np.asarray([r['adv'] for r in rows],dtype=np.float32)
    indices=np.arange(len(rows));rng=np.random.default_rng(SEED+39)
    if mode=='shuffled':
      grouped=collections.defaultdict(list)
      for i,r in enumerate(rows):grouped[(r['family'],r['arm'],r['drop'])].append(i)
      for ix in grouped.values():
        ix=np.asarray(ix,dtype=int);adv[ix]=adv[rng.permutation(ix)]
    targets=1/(1+np.exp(-adv/2.))
    sample_w=.25+np.minimum(3.,np.abs(adv)/2.)
    logs=[]
    for step in range(UPDATES):
      ix=rng.choice(indices,size=BATCH,replace=len(rows)<BATCH)
      logits=head(torch.from_numpy(xs[ix])).squeeze(-1)
      targets_t=torch.from_numpy(targets[ix]);wt=torch.from_numpy(sample_w[ix])
      cls=torch.nn.functional.binary_cross_entropy_with_logits(logits,targets_t,reduction='none')
      loss=(cls*wt).mean()+.015*torch.mean(logits.square())
      opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
      if step in (0,31,63,95,127):
        logs.append({'update':step+1,'loss':float(loss.detach()),
                     'weight_l2':float(torch.linalg.vector_norm(head.weight).detach())})
    return head,{'mode':mode,'updates':UPDATES,'minibatch':BATCH,'rows':len(rows),
        'positive_adv':int(sum(adv>0)),'negative_adv':int(sum(adv<0)),
        'equal_adv':int(sum(adv==0)),'log':logs}


def gate(st,arm,drop,weights):
    x=observe_features(st,arm,drop)
    p=1/(1+np.exp(-float(np.dot(weights,x))))
    return float(np.clip(p,.1,.9))


def choice_uniform(graph,arm,drop):
    # Common uniform random draw for all gate/control policies, independent of
    # whichever model is being evaluated; avoids different-policy RNG confound.
    b=hashlib.sha256(f'C63/{SEED}/{graph}/{arm}/{drop}'.encode()).digest()
    return int.from_bytes(b[:8],'big')/(1<<64)


def evaluate(base,opt,test,weights):
    records=[];traces=[];selected=set()
    for st in test:
      graph=st[6]['graph_sha'];fam=st[8]
      for drop in DROPS:
        for arm in ARMS:
          outcomes={};choices={};gates={}
          # Exact deterministic counterfactual replay: each distinct host-policy
          # suffix is executed ONCE per state/arm/drop. Gate arms select the
          # frozen executed suffix, never interpolate or invent outcomes.
          actuals={}
          for concrete in ('C18-A0','C34-RESET','C60-FRONTIER','C60-ALIGNED'):
            w=weights['C60-FROZEN'] if concrete=='C60-ALIGNED' else np.zeros(c60.NFEAT,np.float32)
            z,_=c60.execute(base,opt,st,arm,concrete,w,drop)
            assert z['host_rejections']==0 and z['protected_sha']==st[3]['protected_sha']
            actuals[concrete]=z
          for pol in POLICIES:
            actual=pol;prob=None
            if pol=='C60-FROZEN':actual='C60-ALIGNED'
            if pol.startswith('C63-'):
              prob=.5 if pol=='C63-COIN' else gate(st,arm,drop,weights[pol])
              actual='C60-FRONTIER' if (arm[0]!='none' and choice_uniform(graph,arm[1],drop)<prob) else 'C34-RESET'
            if arm[0]=='none' and pol.startswith('C63-'):actual='C34-RESET'
            outcomes[pol]=actuals[actual]
            choices[pol]=actual
            if prob is not None:gates[pol]=prob
          if arm[0]=='none':
            assert len({(v['action_sha'],v['ink_sha']) for v in outcomes.values()})==1
          rec={'graph':graph,'family':fam,'case':st[9],'seed':st[7],
             'personality':st[0].personality,'target':list(st[5]),
             'prefix_sha':st[3]['prefix_sha'],'arm':list(arm),'drop':drop,
             'choices':choices,'gate_probabilities':gates,'outcomes':outcomes}
          records.append(rec)
          # Predeclared selection by first TWO eligible graphs in each family,
          # strong cue, intact memory. Re-executed paths independently checked.
          if arm==ARMS[-1] and drop==0. and sum(k[0]==fam for k in selected)<2:
            selected.add((fam,graph))
            for pol in ('C34-RESET','C60-FRONTIER','C63-ALIGNED','C63-SHUFFLED'):
              actual=choices[pol]
              z,_=c60.execute(base,opt,st,arm,actual,weights.get(pol,weights.get('C60-FROZEN')),drop,trace=True)
              assert z['action_sha']==outcomes[pol]['action_sha'] and z['ink_sha']==outcomes[pol]['ink_sha']
              traces.append({'graph':graph,'family':fam,'policy':pol,'actual':actual,
                'start':z['start_pos'],'target':z['target'],'goal':z['goal'],
                'path':z['path'],'action_sha':z['action_sha'],'ink_sha':z['ink_sha']})
    return records,traces


def summarize(records,policies=POLICIES):
    groups=collections.defaultdict(list);personality=collections.defaultdict(list)
    for r in records:
      if r['arm'][0]=='none':continue
      groups[(r['family'],r['drop'])].append(r)
      if r['drop']==0:
        for pol in policies:
          a=r['outcomes'][pol]['personality20'];b=r['outcomes']['C34-RESET']['personality20']
          if a and b:
            personality[pol].append({k:a[k]-b[k] for k in
                  ('turn_rate','reverse_rate','unique_ratio','direction_entropy','paint_fraction')})
    ag={}
    for (fam,drop),rs in groups.items():
      ag[f'{fam}|{drop}']={'episodes':len(rs),'policies':{pol:{
          'goals':sum(r['outcomes'][pol]['goal'] for r in rs),
          'new_sites':sum(r['outcomes'][pol]['new_sites'] for r in rs),
          'paired_wins_vs_RESET':sum(r['outcomes'][pol]['goal']>r['outcomes']['C34-RESET']['goal'] for r in rs),
          'paired_losses_vs_RESET':sum(r['outcomes'][pol]['goal']<r['outcomes']['C34-RESET']['goal'] for r in rs),
          'frontier_gate_rate':sum(r['choices'][pol]=='C60-FRONTIER' for r in rs)/len(rs) if pol.startswith('C63-') else None
        } for pol in policies}}
    pm={}
    for pol,v in personality.items():
      pm[pol]={'matched20':len(v),'mean_deltas':{k:float(np.mean([d[k] for d in v]))
               for k in v[0]} if v else {}}
    return ag,pm


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    c60.SEED=SEED;c60.HORIZON=HORIZON;c60.c59.c58.c55.TICKS=(9,)
    base,opt,_=c60.c59.c58.c55.c54.c52.c50.c47.c44.prerequisites(ROOT/'frozen/predecessor_C42')
    train,test,receipts,exclude=history_and_cohorts()
    statefn=c60.c59.c58.c56.states_for_goal
    trstates,trskip=statefn(base,train,'remote')
    teststates,teskip=statefn(base,test,'remote')
    assert len(trstates)>=30 and len(teststates)>=30
    assert not {s[6]['graph_sha'] for s in trstates}&{s[6]['graph_sha'] for s in teststates}
    trrows,pair_receipts=pair_training(base,opt,trstates)
    aligned,fitlog=fit(trrows,'aligned');shuffled,wrong=fit(trrows,'shuffled')
    # Fully independent refits on same predeclared training data.
    aligned2,_=fit(trrows,'aligned');shuffled2,_=fit(trrows,'shuffled')
    assert torch.equal(aligned.weight,aligned2.weight) and torch.equal(shuffled.weight,shuffled2.weight)
    modelweights={'C63-ALIGNED':aligned.weight.detach().numpy().reshape(-1).copy(),
      'C63-SHUFFLED':shuffled.weight.detach().numpy().reshape(-1).copy()}
    old=json.loads((ROOT/'publication/C60-summary.json').read_text())
    oldw=np.asarray(old['weights']['C60-ALIGNED'],np.float32)
    weights={**modelweights,'C60-FROZEN':oldw}
    cp={}
    for pol,mdl in (('C63-ALIGNED',aligned),('C63-SHUFFLED',shuffled)):
      file=output/(pol+'.safetensors');save_file(mdl.state_dict(),str(file))
      cp[pol]=sha256_bytes(file.read_bytes())
    records,traces=evaluate(base,opt,teststates,weights)
    aggregate,personality=summarize(records)
    unguided=[r for r in records if r['arm'][0]=='none']
    outcome_wins={p:sum(r['outcomes'][p]['goal']>r['outcomes']['C34-RESET']['goal'] for r in records if r['arm'][0]!='none') for p in POLICIES}
    summary={'schema':'driv3r-c63-counterfactual-guidance-gate/1','seed':SEED,'horizon':HORIZON,
      'sources':source_hashes(),'source_sha256':sha256_bytes(Path(__file__).read_bytes()),
      'prior_C60_checkpoint_sha256':old['checkpoint_sha256']['ALIGNED'],'checkpoints':cp,
      'exclusions':exclude,'graph_receipts':receipts,'training_skips':trskip,'test_skips':teskip,
      'training_graphs':len(train),'test_graphs':len(test),'eligible_training':len(trstates),
      'eligible_test':len(teststates),'training_pairs':len(pair_receipts),
      'train_advantages':{'positive':sum(r['adv']>0 for r in trrows),
        'negative':sum(r['adv']<0 for r in trrows),'zero':sum(r['adv']==0 for r in trrows)},
      'optimizer':{'aligned':fitlog,'shuffled':wrong},
      'weights':{k:v.tolist() for k,v in weights.items()},
      'matched_condition_rows':len(records),'policy_outcomes':len(records)*len(POLICIES),
      'independent_host_suffixes':len(records)*4,
      'unguided_parity':len(unguided),'replayed_traces':len(traces),
      'aggregate':aggregate,'movement20':personality,'paired_goal_wins':outcome_wins,
      'limitations':['Synthetic-host-only; no native CyberSand ABI qualification',
      'Episode selector can switch only between frozen RESET and FRONTIER, not learn new spatial options',
      'Counterfactual labels use training-only executed outcomes and are not inference features',
      'Non-guidance parity is a safety control, not full guided artistic-personality preservation',
      'Descriptive metrics; no confirmatory multiple-testing inference',
      'Model binaries retained in archive, not Git or immutable Actions artifacts']}
    full={'schema':'driv3r-c63-full-executed-ledger/1','summary':summary,
          'training_counterfactuals':pair_receipts,'records':records,'gallery_traces':traces}
    raw=json.dumps(full,sort_keys=True,separators=(',',':')).encode()
    (output/'C63-executed-ledger.zlib.b64').write_text(base64.b64encode(zlib.compress(raw,9)).decode()+'\n')
    (output/'C63-summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (output/'C63-traces.json').write_text(json.dumps(traces,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps({'train_graphs':len(train),'test_graphs':len(test),
      'train_eligible':len(trstates),'test_eligible':len(teststates),
      'training_pairs':len(trrows),'optimizer_updates':[fitlog['updates'],wrong['updates']],
      'executed_host_suffixes':len(records)*4,'policy_outcomes':len(records)*len(POLICIES),'traces':len(traces),
      'results':{k:{p:v['goals'] for p,v in d['policies'].items()} for k,d in aggregate.items()}},indent=2))
    return summary

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('out'); args=ap.parse_args();run(args.out)
