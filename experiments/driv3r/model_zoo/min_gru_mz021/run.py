"""MZ-021: 20M minGRU-family scratch policy on the common MZ-003 synthetic host.

Research port of the minGRU recurrence (Feng et al. 2024), not imported
third-party code, not pretrained weights, and NOT CyberSand native gameplay.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, resource, statistics, time
from pathlib import Path
import torch
from torch import nn
from experiments.driv3r.model_zoo.mamba130.world import ACTIONS, World
from experiments.driv3r.model_zoo.mamba130.counterfactual import raw_features, directional_goals, target

UPSTREAM="lucidrains/mingru-pytorch@9fe95d623b2a30f5cbc689e4640dc62403da0df5"
IN_FEATURES=114

class MinGRUBlock(nn.Module):
    def __init__(self, width=800, feedforward=2400):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.gate=nn.Linear(width,width)
        self.candidate=nn.Linear(width,width)
        self.ffnorm=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,feedforward),nn.GELU(),nn.Linear(feedforward,width))
    def step(self, x, state):
        n=self.norm(x)
        z=torch.sigmoid(self.gate(n))
        h=(1-z)*state+z*self.candidate(n)
        y=x+0.5*h
        return y+0.25*self.ff(self.ffnorm(y)),h

class Policy(nn.Module):
    def __init__(self,width=800,layers=4,ff=2400):
        super().__init__()
        self.input=nn.Sequential(nn.Linear(IN_FEATURES,width),nn.LayerNorm(width))
        self.blocks=nn.ModuleList([MinGRUBlock(width,ff) for _ in range(layers)])
        self.readout=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,len(ACTIONS)))
    def initial(self,batch,device="cpu"):
        return tuple(torch.zeros(batch,self.blocks[0].gate.out_features,device=device) for _ in self.blocks)
    def step(self,feature,state):
        x=self.input(feature)
        new=[]
        for block,h in zip(self.blocks,state):
            x,h=block.step(x,h);new.append(h)
        return self.readout(x),tuple(new)
    def forward(self,features,state=None):
        if state is None:state=self.initial(features.size(0),features.device)
        outputs=[]
        for t in range(features.size(1)):
            logits,state=self.step(features[:,t,:],state)
            outputs.append(logits)
        return torch.stack(outputs,dim=1),state

def make_sequence(seed,steps):
    w=World.make(seed);rng=random.Random(seed*101+17);rows=[]
    for t in range(steps):
        original=w.goal
        for goal in directional_goals(w):
            w.goal=goal
            label=target(w)
            rows.append((raw_features(w.prompt()),ACTIONS.index(label),tuple(w.legal()),(seed,t)))
        w.goal=original
        move=w.teacher(rng)
        if move=="STOP":break
        w.apply(move)
    return rows

def all_data(seeds,steps):
    return {str(seed):make_sequence(seed,steps) for seed in seeds}

def tensor_rows(rows):
    return torch.tensor([r[0] for r in rows],dtype=torch.float32),torch.tensor([r[1] for r in rows],dtype=torch.long)

def teacher_metrics(data,predictions):
    total=correct=illegal=forks=fork_both=flips=0
    for seed,rows in data.items():
        preds=predictions[seed]
        groups={}
        for row,p in zip(rows,preds):
            total+=1;correct+=int(p==row[1]);illegal+=int(ACTIONS[p] not in row[2])
            groups.setdefault(row[3],[]).append((row[1],p))
        for g in groups.values():
            for i in range(len(g)):
                for j in range(i+1,len(g)):
                    if g[i][0]!=g[j][0]:
                        forks+=1;fork_both+=int(g[i][0]==g[i][1] and g[j][0]==g[j][1]);flips+=int(g[i][1]!=g[j][1])
    return dict(actions=total,accuracy=round(correct/max(total,1),5),raw_illegal=illegal,
                raw_illegal_rate=round(illegal/max(total,1),5),changed_goal_pairs=forks,
                both_correct=fork_both,paired_goal_accuracy=round(fork_both/max(forks,1),5),
                predicted_goal_switch_rate=round(flips/max(forks,1),5))

def predict(model,data,short=0):
    model.eval();outputs={}
    with torch.inference_mode():
        for seed,rows in data.items():
            if not rows:outputs[seed]=[];continue
            x,y=tensor_rows(rows)
            if short:
                outputs[seed]=[int(model(x[max(0,i-short+1):i+1][None])[0][0,-1].argmax()) for i in range(len(rows))]
            else:
                outputs[seed]=model(x[None])[0][0].argmax(-1).tolist()
    return outputs

def distribution(data):
    counts={a:0 for a in ACTIONS}
    for rows in data.values():
        for row in rows:counts[ACTIONS[row[1]]]+=1
    return counts

def rollouts(model,linear,seeds,horizon=24):
    results=[];latencies=[]
    for seed in seeds:
        for arm in ('mingru','linear','random_nonstop','teacher'):
            w=World.make(seed);r=random.Random(seed*2003+19)
            state=model.initial(1);events=[];raw_illegal=0
            for t in range(horizon):
                menu=w.legal();feature=torch.tensor(raw_features(w.prompt()),dtype=torch.float32)[None]
                if arm in ('mingru','linear'):
                    begin=time.perf_counter()
                    with torch.inference_mode():
                        if arm=='mingru':scores,state=model.step(feature,state)
                        else:scores=linear(feature)
                    if arm=='mingru':latencies.append((time.perf_counter()-begin)*1000)
                    logits=scores[0]
                    raw=ACTIONS[int(logits.argmax())]
                    raw_illegal+=int(raw not in menu)
                    chosen=max(menu,key=lambda a:float(logits[ACTIONS.index(a)]))
                elif arm=='random_nonstop':
                    raw=chosen=r.choice([a for a in menu if a!='STOP'] or ['STOP'])
                else:raw=chosen=w.teacher(r)
                events.append(dict(t=t,position=list(w.pos),raw=raw,selected=chosen,legal=list(menu)))
                w.apply(chosen)
                if w.stopped:break
            results.append(dict(seed=seed,arm=arm,goal=list(w.goal),final=list(w.pos),
                                steps=len(events),painted=len(w.painted),
                                reached_goal=w.pos==w.goal,stopped=w.stopped,
                                final_goal_distance=abs(w.pos[0]-w.goal[0])+abs(w.pos[1]-w.goal[1]),
                                raw_illegal=raw_illegal,events=events))
    return results,latencies

def quantile(values,q):
    return round(sorted(values)[min(len(values)-1,int((len(values)-1)*q))],4) if values else None

def experiment(args):
    if not(1<=args.updates<=200 and 2<=args.train_seeds<=128 and 2<=args.steps<=16):raise ValueError('bounded budget required')
    torch.set_num_threads(args.threads);torch.manual_seed(21021);random.seed(21021)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    train=all_data(range(400,400+args.train_seeds),args.steps)
    dev=all_data(range(900,904),args.steps);test=all_data(range(1200,1204),args.steps)
    assert not(set(train)&set(dev) or set(train)&set(test) or set(dev)&set(test))
    model=Policy();linear=nn.Linear(IN_FEATURES,len(ACTIONS))
    parameters=sum(p.numel() for p in model.parameters())
    assert 20_000_000<=parameters<=22_000_000,parameters
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0004)
    linear_optim=torch.optim.AdamW(linear.parameters(),lr=.004)
    rng=random.Random(21021);losses=[];baseline_losses=[];started=time.monotonic()
    trainkeys=list(train)
    for step in range(args.updates):
        x_batch=[];y_batch=[]
        for _ in range(args.batch):
            rows=train[rng.choice(trainkeys)]
            if len(rows)<args.window:raise RuntimeError('insufficient episode length')
            start=rng.randrange(0,len(rows)-args.window+1)
            x,y=tensor_rows(rows[start:start+args.window]);x_batch.append(x);y_batch.append(y)
        x=torch.stack(x_batch);y=torch.stack(y_batch)
        model.train();optimizer.zero_grad(set_to_none=True)
        logits,_=model(x);loss=nn.functional.cross_entropy(logits.reshape(-1,len(ACTIONS)),y.reshape(-1))
        if not torch.isfinite(loss):raise RuntimeError('nonfinite model loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        linear.train();linear_optim.zero_grad(set_to_none=True)
        base=nn.functional.cross_entropy(linear(x).reshape(-1,len(ACTIONS)),y.reshape(-1))
        base.backward();linear_optim.step()
        losses.append(round(float(loss.detach()),5));baseline_losses.append(round(float(base.detach()),5))
    train_seconds=round(time.monotonic()-started,3)
    model.eval();linear.eval()
    pred=predict(model,test)
    linpred={k:linear(tensor_rows(rows)[0]).argmax(-1).tolist() for k,rows in test.items()}
    shortpred=predict(model,test,short=4)
    devpred=predict(model,dev)
    modelmetrics=teacher_metrics(test,pred);linmetrics=teacher_metrics(test,linpred)
    devmetrics=teacher_metrics(dev,devpred)
    shortmetrics=teacher_metrics(test,shortpred)
    with torch.inference_mode():
        probe=torch.tensor([r[0] for r in next(iter(test.values()))[:10]],dtype=torch.float32)
        full=model(probe[None])[0][0]
        s=model.initial(1);cached=[]
        for token in probe:val,s=model.step(token[None],s);cached.append(val[0])
        cache_error=float((full-torch.stack(cached)).abs().max())
    roll,latencies=rollouts(model,linear,range(1807,1811),args.rollout_steps)
    history_times=[]
    with torch.inference_mode():
        for i in range(24):
            start=time.perf_counter();model(probe[None]);history_times.append((time.perf_counter()-start)*1000)
    checkpoint=out/'min_gru_20m.pt'
    torch.save(dict(model_state=model.state_dict(),linear_state=linear.state_dict(),
                    upstream=UPSTREAM,source_sha=os.getenv('GITHUB_SHA','LOCAL-UNVERIFIED'),
                    train_seeds=list(train),dev_seeds=list(dev),test_seeds=list(test),updates=args.updates,
                    width=800,layers=4,feedforward=2400),checkpoint)
    checksum=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    results=dict(schema='MZ-021-MINGRU-v1',architecture='scratch minGRU recurrence + residual feed-forward',
       source_reference=UPSTREAM,upstream_license='MIT',pretrained=False,source_sha=os.getenv('GITHUB_SHA','LOCAL-UNVERIFIED'),
       model_parameters=parameters,linear_control_parameters=sum(p.numel() for p in linear.parameters()),
       torch_version=torch.__version__,updates=args.updates,training_seconds=train_seconds,
       train_seed_range=[400,400+args.train_seeds-1],dev_seeds=list(map(int,dev)),test_seeds=list(map(int,test)),
       train_sample_count=sum(map(len,train.values())),train_class_counts=distribution(train),
       train_loss_first_last=[losses[0],losses[-1]],baseline_loss_first_last=[baseline_losses[0],baseline_losses[-1]],
       full_train_loss=losses,full_baseline_loss=baseline_losses,
       dev=devmetrics,test=modelmetrics,linear_control=linmetrics,short_context_4=shortmetrics,
       cached_full_max_abs_diff=cache_error,
       rollout_summary={arm:{'reached_goal':sum(r['reached_goal'] for r in roll if r['arm']==arm),
                             'episodes':sum(r['arm']==arm for r in roll),
                             'raw_illegal':sum(r['raw_illegal'] for r in roll if r['arm']==arm),
                             'painted_sum':sum(r['painted'] for r in roll if r['arm']==arm)} for arm in ('mingru','linear','random_nonstop','teacher')},
       cached_cpu_p50_ms=quantile(latencies,.5),cached_cpu_p95_ms=quantile(latencies,.95),
       history10_p50_ms=quantile(history_times,.5),
       whole_process_peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2),
       checkpoint_sha256=checksum,checkpoint_bytes=checkpoint.stat().st_size,
       checkpoint_artifact=None,checkpoint_status='LOCAL_ONLY_NOT_IMMUTABLE',
       warnings=['Not pretrained. Full scratch model training, not an upstream checkpoint.',
                 'MZ-003 world and 114-feature parser; external synthetic world, not native CyberSand.',
                 'Four TEST worlds and 4 rollout worlds give very weak statistical power.',
                 'Host-masked legal execution does not prove raw-policy legality.',
                 'minGRU recurrence adapted independently from the published equations; no associative-scan implementation.',
                 'Training window and test prefix lengths differ; no compute-matched pretrained family comparator.',
                 'Local evaluation: no exact-HEAD independent CI or immutable Actions checkpoint.'])
    (out/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    (out/'trajectories.json').write_text(json.dumps(roll,indent=2)+'\n')
    print(json.dumps({k:results[k] for k in ('model_parameters','updates','training_seconds','test','linear_control','short_context_4','rollout_summary','cached_cpu_p50_ms','cached_cpu_p95_ms','history10_p50_ms','whole_process_peak_rss_mib','checkpoint_sha256','checkpoint_bytes')},indent=2))
    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True);p.add_argument('--updates',type=int,default=64)
    p.add_argument('--train-seeds',type=int,default=24);p.add_argument('--steps',type=int,default=8)
    p.add_argument('--window',type=int,default=8);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--rollout-steps',type=int,default=24)
    experiment(p.parse_args())
if __name__=='__main__':main()
