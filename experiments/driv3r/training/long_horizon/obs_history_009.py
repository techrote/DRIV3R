"""DRIV3R OBS-HISTORY-009. Opt-in CPU synthetic policy trial; NOT CyberSand ABI.
Usage: python obs_history_009.py /absolute/output/directory
No structural HELDOUT, network or paid inference. No automatic training on import.
"""
import copy
import hashlib
import json
import math
import os
import resource
import statistics
import sys
import time
from collections import Counter, deque

HORIZONS = (36, 72, 128)
TRAIN_SEEDS = range(1100, 1220)
DEV_SEEDS = range(3100, 3140)
MOVES = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}
ACTIONS = ('N','E','S','W','PAINT','ERASE','STOP')
POLICIES = ('never_recheck','always_recheck','observed_warning','learned_history','learned_current','privileged_oracle')
TARGETS = ((2,1),(10,1),(10,7),(2,7),(10,3))


def hash_obj(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def rand(seed, tag):
    return int(hashlib.sha256(f'{seed}/{tag}'.encode()).hexdigest()[:12],16) % 10000


def grid(seed, split):
    y1=(1,3,5,7)[rand(seed,'a')%4]
    y2=(1,3,5,7)[rand(seed,'b')%4]
    walls={(4,y) for y in range(9) if y!=y1} | {(8,y) for y in range(9) if y!=y2}
    if split=='dev':
        walls|={(6,0)} if seed%3==0 else set()
        walls|={(6,8)} if seed%4==0 else set()
    return walls


def route(start,goal,walls):
    q=deque([start]); last={start:None}; actions={}
    while q:
        p=q.popleft()
        if p==goal:break
        for act,(dx,dy) in MOVES.items():
            n=(p[0]+dx,p[1]+dy)
            if 0<=n[0]<13 and 0<=n[1]<9 and n not in walls and n not in last:
                last[n]=p;actions[n]=act;q.append(n)
    if goal not in last:return []
    out=[];p=goal
    while p!=start:
        out.append(actions[p]);p=last[p]
    return out[::-1]


class Host:
    def __init__(self,seed,horizon,split):
        self.seed,self.horizon,self.split=seed,horizon,split
        self.walls=grid(seed,split)
        self.pos=(1,1);self.ink={};self.tick=0;self.last=None
        self.log=[];self.messages=[];self.pending=[];self.planner=0;self.backtracks=0
        self.event_done=False;self.changed=False;self.fault_done=False;self.fault_target=rand(seed,'fault')%3
        self.temp_cell=None;self.temp_until=0;self.last_paint_first=None
        self.n={36:2,72:3,128:5}[horizon]
        self.event_at=max(7,horizon//4)
    def obstacles(self):
        b=set(self.walls)
        if self.temp_cell is not None and self.tick<self.temp_until:b.add(self.temp_cell)
        return b
    def menu(self):
        b=self.obstacles();out=[]
        for a,(dx,dy) in MOVES.items():
            t=(self.pos[0]+dx,self.pos[1]+dy)
            if 0<=t[0]<13 and 0<=t[1]<9 and t not in b:out.append(a)
        if self.pos not in {(6,0),(6,8)} and self.pos not in b:
            out.append('ERASE' if self.pos in self.ink else 'PAINT')
        out.append('STOP');return out
    def visible(self):
        return self.ink.get(self.pos)
    def step(self,a):
        legal=self.menu();assert a in legal,(self.seed,self.tick,a,legal)
        before=self.pos;before_ink=self.visible();result='ok'
        if a in MOVES:
            dx,dy=MOVES[a];self.pos=(self.pos[0]+dx,self.pos[1]+dy)
        elif a=='PAINT':
            if (self.pos==TARGETS[self.fault_target]) and not self.fault_done:
                self.ink[self.pos]=2;self.fault_done=True;result='wrong_material'
            else:self.ink[self.pos]=1;result='painted'
            if self.pos==TARGETS[0] and result=='painted':self.last_paint_first=self.tick
        elif a=='ERASE':self.ink.pop(self.pos);result='erased'
        elif a=='STOP':result='stopped'
        receipt={'tick':self.tick,'action':a,'legal_menu':legal,'before':before,'after':self.pos,
                 'visible_before':before_ink,'result':result,'visible_after':self.visible()}
        self.log.append(receipt);self.last=receipt;self.tick+=1
        if not self.event_done and self.tick>=self.event_at and self.last_paint_first is not None:
            self.event_done=True
            self.changed=rand(self.seed,'change')%10<6
            if self.changed:self.ink.pop(TARGETS[0],None)
            noisy_warning=(rand(self.seed,'hint')%10<7) if self.changed else (rand(self.seed,'false')%10<3)
            delay=(0,3,7,None,10)[rand(self.seed,'delay')%5]
            if delay is not None:
                self.pending.append({'delivery_tick':self.tick+delay,'kind':'warning' if noisy_warning else 'epoch',
                                     'source_tick':self.tick})
            # All observed information is limited to receipt stream, locally visible ink and message kind.
        if self.tick==max(10,self.horizon//3):
            cell=(6,(1,3,5,7)[rand(self.seed,'gate')%4])
            if cell not in self.walls and cell!=self.pos:
                self.temp_cell=cell;self.temp_until=self.tick+3+rand(self.seed,'duration')%4
        delivered=[m for m in self.pending if m['delivery_tick']<=self.tick]
        self.pending=[m for m in self.pending if m['delivery_tick']>self.tick]
        for m in delivered:self.messages.append({'seen_tick':self.tick, 'kind':m['kind'], 'source_tick':m['source_tick']})
        return receipt
    def goal(self):
        return all(self.ink.get(p)==1 for p in TARGETS[:self.n])
    def outcome(self,cap):
        return {'goals':int(self.goal()), 'ticks':self.tick, 'planner_calls':self.planner,
                'backtracks':self.backtracks,'utility':1000*int(self.goal())-2*self.tick-3*self.planner-self.backtracks,
                'counts':dict(Counter(r['action'] for r in self.log)),
                'receipt_sha256':hash_obj(self.log), 'final_ink_sha256':hash_obj(sorted((x,y,c) for (x,y),c in self.ink.items())),
                'positions':[r['after'] for r in self.log], 'cap':cap}


def travel(host,goal,cap):
    path=[];dest=None
    while host.pos!=goal and host.tick<cap:
        if dest!=goal or not path or path[0] not in host.menu():
            path=route(host.pos,goal,host.obstacles());dest=goal;host.planner+=1
        if not path:return False
        a=path.pop(0)
        if a in ('N','W'):host.backtracks+=1
        host.step(a)
    return host.pos==goal


def paint_correctly(host,cap):
    if host.pos in host.ink and host.ink[host.pos]!=1 and host.tick<cap:host.step('ERASE')
    if host.pos not in host.ink and host.tick<cap:host.step('PAINT')
    if host.visible()==2 and host.tick<cap:
        host.step('ERASE')
        if host.tick<cap:host.step('PAINT')
    return host.visible()==1


def prefix(seed,horizon,split,cap):
    h=Host(seed,horizon,split)
    for dest in TARGETS[:h.n]:
        if h.tick>=cap or not travel(h,dest,cap):break
        if not paint_correctly(h,cap):break
    return h


def features(h,cap,history):
    # Contemporaneously known variables only. No seed, mutation truth, pending messages or hidden ink.
    warn=[m for m in h.messages if m['kind']=='warning']
    epochs=[m for m in h.messages if m['kind']=='epoch']
    age=(h.tick-h.last_paint_first) if h.last_paint_first is not None else 0
    current=[m for m in h.messages if m['seen_tick']==h.tick]
    tail=(current[-1]['kind'] if current else 'none')
    return [1.0, (cap-h.tick)/128, h.tick/128, min(1.0,age/128) if history else 0.0,
            float(bool(warn)) if history else float(tail=='warning'),
            float(bool(epochs)) if history else float(tail=='epoch'),
            min(1.0,(h.tick-warn[-1]['seen_tick'])/64) if (history and warn) else 0.0,
            min(1.0, len(route(h.pos,TARGETS[0],h.obstacles()))/40),
            float(h.horizon==36),float(h.horizon==72),float(h.horizon==128)]


def finish(h,cap,recheck):
    if recheck and h.tick<cap:
        if travel(h,TARGETS[0],cap):paint_correctly(h,cap)
    if h.tick<cap:h.step('STOP')
    return h.outcome(cap)


def fork(h,cap):
    a=copy.deepcopy(h);b=copy.deepcopy(h)
    return finish(a,cap,False),finish(b,cap,True)


def sigmoid(x):return 1/(1+math.exp(-max(-35,min(35,x))))


def fit(rows,history):
    # Cross-entropy imitation of TRAIN-only *counterfactual utility* winner, with regularization.
    weights=[0.0]*11;examples=[]
    for h,cap,base,checked in rows:
        x=features(h,cap,history)
        target=float(checked['utility']>base['utility'])
        examples.append((x,target))
    for step in range(1600):
        grad=[0.0]*len(weights)
        for x,target in examples:
            err=sigmoid(sum(w*v for w,v in zip(weights,x)))-target
            for i,v in enumerate(x):grad[i]+=err*v
        rate=0.6/(1+step/3000)
        for i in range(len(weights)):
            weights[i]-=rate*(grad[i]/len(examples)+0.0003*weights[i])
    agreement=sum((sigmoid(sum(w*v for w,v in zip(weights,x)))>=.5)==bool(y) for x,y in examples)
    return {'weights':[round(w,12) for w in weights],'steps':1600,'examples':len(examples),
            'label_positives':sum(y for x,y in examples),'train_agreement':round(agreement/len(examples),6)}


def decision(policy,h,cap,models):
    if policy=='never_recheck':return False
    if policy=='always_recheck':return True
    if policy=='observed_warning':return any(m['kind']=='warning' for m in h.messages)
    if policy=='privileged_oracle':
        b,c=fork(h,cap);return c['utility']>b['utility']
    history=policy=='learned_history'; m=models[policy]
    x=features(h,cap,history)
    return sigmoid(sum(w*v for w,v in zip(m['weights'],x)))>=0.5


def audit_receipts(h):
    # Separate algebraic checks; same-source Host replay is also applied below.
    for i,r in enumerate(h.log):
        assert r['tick']==i and r['action'] in r['legal_menu']
        x,y=r['before'];u,v=r['after'];a=r['action']
        if a in MOVES:
            dx,dy=MOVES[a];assert (u,v)==(x+dx,y+dy)
        else:assert (u,v)==(x,y)
        if a=='ERASE':assert r['visible_after'] is None
        if a=='PAINT':assert r['visible_after'] in (1,2)
    replay=Host(h.seed,h.horizon,h.split)
    for r in h.log:assert replay.step(r['action'])==r
    assert replay.ink==h.ink and replay.pos==h.pos and replay.messages==h.messages
    return len(h.log)


def campaign():
    started=time.perf_counter(); train=[]
    for seed in TRAIN_SEEDS:
        for horizon in HORIZONS:
            for cap in ([horizon] if horizon==36 else [64,128] if horizon==128 else [64]):
                h=prefix(seed,horizon,'train',cap)
                a,b=fork(h,cap)
                train.append((h,cap,a,b))
    models={'learned_history':fit(train,True),'learned_current':fit(train,False)}
    rows=[]; cases=[]; all_lat=[]; action_count=0; tamper=False
    for seed in DEV_SEEDS:
        for horizon in HORIZONS:
            for cap in ([36] if horizon==36 else [64,128] if horizon==128 else [64]):
                start=prefix(seed,horizon,'dev',cap)
                for policy in POLICIES:
                    h=copy.deepcopy(start)
                    t=time.perf_counter_ns();chosen=decision(policy,h,cap,models)
                    all_lat.append((policy,time.perf_counter_ns()-t))
                    outcome=finish(h,cap,chosen)
                    action_count+=audit_receipts(h)
                    # Deterministic full source replay/receipt hash via a second fork.
                    r=finish(copy.deepcopy(start),cap,chosen)
                    assert (r['receipt_sha256'],r['utility'])==(outcome['receipt_sha256'],outcome['utility'])
                    row={k:v for k,v in outcome.items() if k!='positions'}
                    row.update({'policy':policy,'seed':seed,'horizon':horizon,'recheck':chosen,
                                'cap_kind':'shadow128' if cap==128 else 'native64_proxy',
                                'observed_messages':start.messages,'changed_hidden_for_policy':start.changed,
                                'prefix_ticks':start.tick,'features':features(start,cap,policy=='learned_history') if policy.startswith('learned') else []})
                    rows.append(row)
                    if seed == 3107 and policy in ('learned_history','always_recheck','never_recheck'):
                        cases.append({'seed':seed,'horizon':horizon,'cap':cap,'policy':policy,'goal':outcome['goals'],
                                      'recheck':chosen,'positions':outcome['positions'], 'receipt_sha256':outcome['receipt_sha256']})
    bad=[dict(x) for x in h.log]
    bad[0]['action']='TELEPORT'
    try:
        assert bad[0]['action'] in bad[0]['legal_menu']
    except AssertionError:tamper=True
    assert tamper
    results={}
    for policy in POLICIES:
        r=[x for x in rows if x['policy']==policy]
        dist=Counter()
        for x in r:dist.update(x['counts'])
        results[policy]={'episodes':len(r),'goals':sum(x['goals'] for x in r),
                         'mean_utility':round(statistics.mean(x['utility'] for x in r),4),
                         'rechecks':sum(x['recheck'] for x in r),
                         'planner_calls':sum(x['planner_calls'] for x in r),
                         'action_distribution':{a:dist[a] for a in ACTIONS}}
    base={(r['seed'],r['horizon'],r['cap']):r for r in rows if r['policy']=='always_recheck'}
    paired={}
    for policy in POLICIES:
        if policy=='always_recheck':continue
        s=[r for r in rows if r['policy']==policy]; wins=[]; losses=[];ties=0
        for r in s:
            b=base[(r['seed'],r['horizon'],r['cap'])];d=r['utility']-b['utility']
            example={'seed':r['seed'],'horizon':r['horizon'],'cap':r['cap'],'delta':d,
                     'policy_goal':r['goals'],'baseline_goal':b['goals'],
                     'receipt_sha256':r['receipt_sha256']}
            if d>0:wins.append(example)
            elif d<0:losses.append(example)
            else:ties+=1
        paired[policy]={'wins':len(wins),'losses':len(losses),'ties':ties,'negative_cases':losses,
                        'positive_cases':wins[:8]}
    cohorts={}
    for h,c in ((36,36),(72,64),(128,64),(128,128)):
        cohorts[f'{h}/{c}']={p:sum(r['goals'] for r in rows if r['horizon']==h and r['cap']==c and r['policy']==p) for p in POLICIES}
    decision_lat={}
    for p in POLICIES:
        times=sorted(v/1000 for q,v in all_lat if q==p)
        decision_lat[p]={'p50_us':round(times[len(times)//2],3),'p95_us':round(times[int(.95*(len(times)-1))],3),
                         'n':len(times)}
    evidence={'study':'OBS-HISTORY-009','date':'2026-11-19','host':'synthetic spatial/material world; NOT native ABI',
              'design':'Training-only counterfactual utility teacher; evaluation policies receive observable message history, local ink and budget, never latent material-change truth. Privileged oracle explicitly labelled.',
              'train_seeds':[min(TRAIN_SEEDS),max(TRAIN_SEEDS)],'dev_seeds':[min(DEV_SEEDS),max(DEV_SEEDS)],
              'sealed_structural_heldout_accessed':False,'optimizer':models,
              'summary':results,'paired_vs_always_recheck':paired,'cohorts':cohorts,
              'episodes':len(rows),'replayed_action_receipts':action_count,
              'latency_microseconds':decision_lat,'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'runtime_seconds':round(time.perf_counter()-started,3),'tamper_rejected':tamper,
              'representative_paths':cases,'limitations':[
                  'Simulation uses synthetic receipts and host semantics, not verified CyberSand ABI.',
                  'The structural HELDOUT is unopened; DEV results only.',
                  'Teacher labels use TRAIN-only privileged counterfactual utility; policies cannot observe latent erasure.',
                  'Replay is same-source Host plus independent receipt algebra, not independent native replay.',
                  '64 logical moves are a synthetic censor proxy; 128 shadow trials are not native game completion.',
                  'Latency is Python dispatch on this CPU, not inference with a deployed LFM/d1/Sprocket checkpoint.']}
    assert len(train)==480 and len(rows)==960 and action_count>10000
    assert len(all_lat)==960
    return evidence


def gallery(e):
    cases=e['representative_paths'];parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="750" height="{112*len(cases)+30}" viewBox="0 0 750 {112*len(cases)+30}">',
                                    '<rect width="100%" height="100%" fill="#15202a"/>']
    for i,c in enumerate(cases):
        y=20+i*112
        pts=' '.join(f'{50+x*18},{y+12+yy*9}' for x,yy in c['positions'])
        parts.append(f'<text x="360" y="{y+16}" fill="#f7f7f7" font-family="monospace" font-size="12">{c["policy"]} h{c["horizon"]} cap{c["cap"]} seed{c["seed"]} goal{c["goal"]}</text>')
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{["#8dd4f0","#e1b567","#c4a8e7"][i%3]}" stroke-width="1.6"/>')
    return '\n'.join(parts+['</svg>'])+'\n'

if __name__=='__main__':
    if len(sys.argv)!=2 or not os.path.isabs(sys.argv[1]):raise SystemExit('Explicit opt-in: python obs_history_009.py /absolute/output/directory')
    out=sys.argv[1];os.makedirs(out,exist_ok=True)
    e=campaign();e['source_sha256']=hashlib.sha256(open(__file__,'rb').read()).hexdigest()
    with open(os.path.join(out,'OBS_HISTORY_009_EVIDENCE.json'),'w') as f:json.dump(e,f,indent=2);f.write('\n')
    with open(os.path.join(out,'OBS_HISTORY_009_PATHS.svg'),'w') as f:f.write(gallery(e))
    print(json.dumps({k:e[k] for k in ('summary','cohorts','episodes','replayed_action_receipts','maxrss_kib','runtime_seconds','tamper_rejected')},indent=2))
