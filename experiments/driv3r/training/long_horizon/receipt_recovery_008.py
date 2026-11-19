"""RECEIPT-RECOVERY-008: explicit opt-in CPU-only synthetic game-policy experiment.
Run: python receipt_recovery_008.py /absolute/output/dir
NOT the CyberSand host ABI. Structural HELDOUT is neither read nor generated.
"""
import hashlib, json, math, os, resource, sys, time
from collections import Counter, deque

HORIZONS = (36, 72, 128)
TRAIN = tuple(range(1100, 1164))
DEV = tuple(range(3100, 3124))
ACTIONS = ('N','E','S','W','PAINT','ERASE','STOP')
DIRECTIONS = [('N',(0,-1)),('E',(1,0)),('S',(0,1)),('W',(-1,0))]

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def landscape(seed, split):
    w,h=13,9
    gap1=(1,3,5,7)[seed%4]
    gap2=(1,3,5,7)[(seed//4+ (1 if split=='dev' else 0))%4]
    blocked={(x,y) for x,g in ((4,gap1),(8,gap2)) for y in range(h) if y!=g}
    # Deepening evaluation topology: a passage shift, never peeked at during fitting.
    if split=='dev' and seed%3==0:
        blocked|={(6,2),(6,3)}
    targets=((2,1),(11,1),(11,7),(1,7),(1,1),(11,4),(2,7),(10,7))
    return {'width':w,'height':h,'blocked':blocked,'protected':{(6,0),(6,8)},'targets':targets}

def bfs(grid,blocked,start,goal):
    q=deque([start]); parent={start:None}; step={}
    while q:
        p=q.popleft()
        if p==goal:break
        for name,(dx,dy) in DIRECTIONS:
            n=(p[0]+dx,p[1]+dy)
            if 0<=n[0]<grid['width'] and 0<=n[1]<grid['height'] and n not in blocked and n not in parent:
                parent[n]=p;step[n]=name;q.append(n)
    if goal not in parent:return []
    path=[];p=goal
    while p!=start:path.append(step[p]);p=parent[p]
    return path[::-1]

class Host:
    def __init__(self,seed,horizon,split):
        self.seed,self.horizon,self.split=seed,horizon,split
        self.grid=landscape(seed,split);self.pos=(1,1);self.ink={};self.tick=0
        self.stage=0;self.events=[];self.last=None;self.mutation_done=False
        self.fault_stage=seed%3;self.fault_done=False
        self.move_count=0;self.stops=0;self.last_mutation=None
        self.temp_until=0;self.temp_cell=None
    def blocked(self):
        b=set(self.grid['blocked'])
        if self.temp_cell is not None and self.tick<self.temp_until:b.add(self.temp_cell)
        return b
    def observation(self):
        # Synthetic host-style local state + action receipt. No privileged future event schedule.
        return {'position':self.pos,'stage':self.stage,'last':self.last,
                'material_epoch':len(self.events),'visible_ink':self.ink.get(self.pos),
                'remaining':self.horizon-self.tick,'signal':self.events[-1] if self.events and self.events[-1]['tick']==self.tick else None}
    def event(self):
        self.tick+=1
        # A geometry-altering material pulse requires replanning only if it intersects the active route.
        if self.tick==max(8,self.horizon//3):
            y=(1,3,5,7)[(self.seed+1)%4]
            cell=(6,y)
            if cell not in self.grid['blocked'] and cell!=self.pos:
                self.temp_cell=cell;self.temp_until=self.tick+3+(self.seed%4)
        # Material epoch can change without invalidating any painted target.
        if not self.mutation_done and self.stage>=2 and self.tick>=self.horizon//2:
            self.mutation_done=True
            changed=self.seed%4!=0
            target=self.grid['targets'][0]
            if changed:self.ink.pop(target,None)
            # 75% of worlds genuinely mutate a prior paint. Receipt exposes only the observed event.
            self.events.append({'tick':self.tick,'type':'material_change' if changed else 'epoch_only',
                                'target':target if changed else None})
    def legal(self):
        menu=[];blocked=self.blocked()
        for name,(dx,dy) in DIRECTIONS:
            p=(self.pos[0]+dx,self.pos[1]+dy)
            if 0<=p[0]<13 and 0<=p[1]<9 and p not in blocked:menu.append(name)
        if self.pos not in self.grid['protected'] and self.pos not in self.grid['blocked']:
            if self.pos in self.ink:menu.append('ERASE')
            else:menu.append('PAINT')
        menu.append('STOP');return menu
    def step(self,action,desired=1):
        menu=self.legal();assert action in menu,(action,menu)
        before={'pos':self.pos,'ink':sorted((x,y,c) for (x,y),c in self.ink.items()),'stage':self.stage}
        result='ok'
        if action in ('N','E','S','W'):
            dx,dy=dict(DIRECTIONS)[action];self.pos=(self.pos[0]+dx,self.pos[1]+dy);self.move_count+=1
        elif action=='ERASE':self.ink.pop(self.pos,None);result='erased'
        elif action=='PAINT':
            if self.stage==self.fault_stage and not self.fault_done:
                self.ink[self.pos]=2;result='wrong_material';self.fault_done=True
            else:self.ink[self.pos]=desired;result='painted'
        elif action=='STOP':self.stops+=1
        self.last={'action':action,'result':result,'position':self.pos}
        receipt={'tick':self.tick,'action':action,'result':result,'legal_menu':menu,
                 'before_sha256':digest(before),'position':self.pos,'stage':self.stage}
        self.event()
        return receipt
    def complete(self,n):return all(self.ink.get(p)==1 for p in self.grid['targets'][:n])

# Vector includes only contemporaneous state / observed receipt information, never world seed/mode or future events.
def feature(bad,alert,epoch_only,remaining,dist):
    return [1.0,float(bad),float(alert),float(epoch_only),min(1.0,remaining/128),min(1.0,dist/24)]

def sigmoid(z):return 1/(1+math.exp(-max(-30,min(30,z))))

def train(with_feedback):
    samples=[]
    for seed in TRAIN:
        for horizon in HORIZONS:
            for kind in ('bad','changed','epoch','normal'):
                # balanced positive and negative receipt situations; strictly TRAIN generated.
                bad=int(kind=='bad');alert=int(kind=='changed');epoch=int(kind=='epoch')
                x=feature(bad if with_feedback else 0,alert if with_feedback else 0,
                          epoch if with_feedback else 0,horizon-(seed%horizon),seed%17)
                samples.append((x,float(bool(bad or alert))))
    weights=[0.0]*6
    for _ in range(1000):
        grad=[0.0]*6
        for x,y in samples:
            d=(sigmoid(sum(a*b for a,b in zip(weights,x)))-y)/len(samples)
            for j in range(6):grad[j]+=d*x[j]
        for j in range(6):weights[j]-=0.35*(grad[j]+0.0003*weights[j])
    correct=sum((sigmoid(sum(a*b for a,b in zip(weights,x)))>=0.5)==(y>0.5) for x,y in samples)
    return {'weights':weights,'optimizer_steps':1000,'train_examples':len(samples),'train_agreement':correct/len(samples)}

def repair_decision(policy,model,kind,remaining,dist):
    if policy=='receipt_rule':return kind in ('bad','changed')
    if policy=='blind':return False
    if policy=='always_repair':return True
    if policy in ('learned','no_feedback'):
        x=feature(kind=='bad' if policy=='learned' else False,
                  kind=='changed' if policy=='learned' else False,
                  kind=='epoch' if policy=='learned' else False,remaining,dist)
        return sigmoid(sum(a*b for a,b in zip(model['weights'],x)))>=0.5
    raise ValueError(policy)

def run(seed,horizon,split,cap,policy,models,record=True):
    host=Host(seed,horizon,split)
    n={36:2,72:4,128:7}[horizon];tasks=list(host.grid['targets'][:n]);stage=0
    current=None;route=[];repair=[];counts=Counter();planner_calls=0;backtracks=0
    seen_event=0;wrong_pending=False;trace=[];decisions=[];inference_ns=[]
    while host.tick<cap:
        obs=host.observation()
        if obs['signal'] and len(host.events)>seen_event:
            seen_event=len(host.events)
            signal=obs['signal'];kind='changed' if signal['type']=='material_change' else 'epoch'
            t=time.perf_counter_ns();yes=repair_decision(policy,models.get(policy),kind,obs['remaining'],abs(host.pos[0]-tasks[0][0])+abs(host.pos[1]-tasks[0][1]));inference_ns.append(time.perf_counter_ns()-t)
            decisions.append({'kind':kind,'selected':yes,'tick':host.tick})
            if yes and kind=='changed' and tasks[0] not in repair:repair.append(tasks[0])
            if yes and kind=='epoch' and stage>0 and tasks[0] not in repair:repair.append(tasks[0])
        if host.last and host.last['action']=='PAINT' and host.last['result']=='wrong_material' and not wrong_pending:
            t=time.perf_counter_ns();yes=repair_decision(policy,models.get(policy),'bad',obs['remaining'],0);inference_ns.append(time.perf_counter_ns()-t)
            decisions.append({'kind':'bad','selected':yes,'tick':host.tick})
            if yes:wrong_pending=True
            else:stage+=1
        # Resolve a previously botched paint at current location; never paint on protected cells.
        if wrong_pending:
            action='ERASE' if host.pos in host.ink else 'PAINT'
            if action=='PAINT':wrong_pending=False
        else:
            dest=repair[0] if repair else (tasks[stage] if stage<n else None)
            if dest is None:
                action='STOP'
            elif host.pos==dest:
                c=host.ink.get(dest)
                if c==1:
                    if repair:repair.pop(0)
                    else:stage+=1
                    route=[];current=None;continue
                elif c is not None:action='ERASE'
                else:action='PAINT'
                if action=='PAINT' and repair:repair.pop(0)
            else:
                if current!=dest or not route or route[0] not in host.legal():
                    route=bfs(host.grid,host.blocked(),host.pos,dest);current=dest;planner_calls+=1
                if not route:break
                action=route.pop(0)
                if action in ('W','N'):backtracks+=1
        if action not in host.legal():break
        host.stage=stage
        receipt=host.step(action);counts[action]+=1
        if record:trace.append(receipt)
        if action=='STOP':break
    arrived=host.complete(n)
    # Tighter utility rewards only verified final material objectives, not local 'stage' bookkeeping.
    utility=1000*int(arrived)-2*host.tick-3*planner_calls-backtracks
    return {'seed':seed,'horizon':horizon,'split':split,'cohort':'shadow128' if cap==128 else 'native64_proxy',
            'policy':policy,'cap':cap,'goals':int(arrived),'ticks':host.tick,'paint':counts['PAINT'],'erase':counts['ERASE'],
            'planner_calls':planner_calls,'backtracks':backtracks,'utility':utility,'counts':dict(sorted(counts.items())),
            'checkpoint_events':decisions,'receipts':trace,'receipt_sha256':digest(trace),
            'final_ink_sha256':digest(sorted((x,y,c) for (x,y),c in host.ink.items())),
            'positions':[r['position'] for r in trace],'inference_ns':inference_ns,'stage':stage}

def replay_trace(seed,horizon,split,receipts):
    host=Host(seed,horizon,split)
    for receipt in receipts:
        host.stage=receipt['stage']
        assert host.step(receipt['action'])==receipt,'Host action/receipt replay mismatch'
    return digest(sorted((x,y,c) for (x,y),c in host.ink.items()))

def evidence():
    t0=time.perf_counter();m={'learned':train(True),'no_feedback':train(False)}
    assert m['learned']['train_examples']==768 and m['learned']['train_agreement']>.99
    policies=('receipt_rule','learned','no_feedback','blind','always_repair')
    rows=[];cases=[];latencies=[];verified=0;tamper_checked=False
    for h in HORIZONS:
        for seed in DEV:
            for cap in ((h if h<64 else 64),)+( (128,) if h==128 else () ):
                for p in policies:
                    a=run(seed,h,'dev',cap,p,m)
                    b=run(seed,h,'dev',cap,p,m)
                    assert a['receipt_sha256']==b['receipt_sha256'] and a['final_ink_sha256']==b['final_ink_sha256']
                    assert replay_trace(seed,h,'dev',a['receipts'])==a['final_ink_sha256']
                    assert all(r['action'] in r['legal_menu'] for r in a['receipts'])
                    assert len(a['receipts'])==a['ticks']
                    verified+=len(a['receipts'])
                    latencies+=a['inference_ns']
                    cases.append(a)
                    rows.append({k:v for k,v in a.items() if k not in ('receipts','positions','inference_ns')})
    def group(p):
        rr=[a for a in rows if a['policy']==p]
        return {'episodes':len(rr),'goals':sum(a['goals'] for a in rr),'mean_utility':round(sum(a['utility'] for a in rr)/len(rr),4),
                'total_planner_calls':sum(a['planner_calls'] for a in rr),'total_actions':sum(a['ticks'] for a in rr),
                'action_distribution':dict(Counter({k:sum(a['counts'].get(k,0) for a in rr) for k in ACTIONS}))}
    summary={p:group(p) for p in policies}
    # Pairwise controls, all paired DEV world/horizon/cap scenarios identical.
    by={(a['seed'],a['horizon'],a['cap'],a['policy']):a for a in rows};paired={}
    for p in policies[1:]:
        wins=losses=ties=lost=gained=0;delta=0;neg=[]
        for a in rows:
            if a['policy']!='receipt_rule':continue
            b=by[(a['seed'],a['horizon'],a['cap'],p)];d=b['utility']-a['utility'];delta+=d
            wins+=d>0;losses+=d<0;ties+=d==0;lost+=a['goals']>b['goals'];gained+=a['goals']<b['goals']
            if d<0 and len(neg)<10:neg.append({'seed':a['seed'],'horizon':a['horizon'],'cap':a['cap'],'delta':d,'baseline_goal':a['goals'],'alternative_goal':b['goals'],'alternative_receipt_sha256':b['receipt_sha256']})
        paired[p]={'wins':wins,'losses':losses,'ties':ties,'goals_lost':lost,'goals_gained':gained,'mean_utility_delta':round(delta/sum(a['policy']=='receipt_rule' for a in rows),4),'negative_examples':neg}
    # A manually and independently replayed validation is not claimed; the same-source replay is explicit.
    bad=cases[0]['receipts'].copy();bad[0]=dict(bad[0]);bad[0]['action']='ILLEGAL'
    try:replay_trace(cases[0]['seed'],cases[0]['horizon'],'dev',bad)
    except (AssertionError,ValueError):tamper_checked=True
    assert tamper_checked
    latencies.sort();quant=lambda q:round(latencies[min(len(latencies)-1,int(q*(len(latencies)-1)))]/1000,3) if latencies else None
    evidence={'study':'RECEIPT-RECOVERY-008','date':'2026-11-19','purpose':'Train on observable paint/epoch receipts, test wrong-paint repair and prior-target material-change recovery under DEV topology transfer.',
      'source_contract':'Synthetic, NOT verified native CyberSand ABI','splits':{'train_seeds':[TRAIN[0],TRAIN[-1]],'dev_seeds':[DEV[0],DEV[-1]],'structural_heldout_opened':False},
      'source_sha256':hashlib.sha256(open(__file__,'rb').read()).hexdigest(),
      'training':m,'horizons':HORIZONS,'native64_censor':64,'shadow128_is_synthetic':True,
      'policies':policies,'summary':summary,'paired_vs_receipt_rule':paired,
      'cohorts':{f'{h}/{cap}':{p:sum(a['goals'] for a in rows if a['horizon']==h and a['cap']==cap and a['policy']==p) for p in policies} for h,cap in ((36,36),(72,64),(128,64),(128,128))},
      'synthetic_episode_count':len(rows),
      'same_source_replayed_action_receipts':verified,'tamper_rejected':tamper_checked,
      'inference_microseconds':{'n':len(latencies),'p50':quant(.50),'p95':quant(.95),'p99':quant(.99)},
      'total_wall_seconds':round(time.perf_counter()-t0,3),'process_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'representative_cases':[{'seed':a['seed'],'horizon':a['horizon'],'cap':a['cap'],'policy':a['policy'],'receipt_sha256':a['receipt_sha256'],'positions':a['positions'],'goal':a['goals']} for a in cases if a['seed']==3102 and a['horizon'] in (72,128) and a['policy'] in ('receipt_rule','learned','no_feedback')],
      'limitations':['Receipt replay uses a fresh same-source host without consulting the policy, not an independently implemented host','Synthetic global event receipt is a research assumption, not a production API','No native game validation, GPU, external model adapters or checkpoint binaries','SOURCE-SMOKE is not scientific or native validation']}
    assert len(rows)==480 and len(latencies)>0 and verified>1000
    return evidence

def svg_gallery(e):
    cases=e['representative_cases'];parts=['<svg xmlns="http://www.w3.org/2000/svg" width="840" height="{}" viewBox="0 0 840 {}">'.format(170*len(cases)+40,170*len(cases)+40),'<rect width="100%" height="100%" fill="#101820"/>']
    for i,a in enumerate(cases):
        y=25+i*170;pts=a['positions'];offset=24
        data=' '.join(f'{offset+x*15:.1f},{y+22+yy*13:.1f}' for x,yy in pts)
        label=f"{a['policy']} / h{a['horizon']} / cap{a['cap']} / goal {a['goal']} / seed {a['seed']}"
        parts.extend([f'<text x="22" y="{y}" font-family="monospace" font-size="13" fill="#eef4f8">{label}</text>',
                      f'<polyline points="{data}" fill="none" stroke="{["#86d5ec","#f8be75","#dd85bb"][i%3]}" stroke-width="1.7"/>',
                      f'<circle cx="{offset+pts[0][0]*15}" cy="{y+22+pts[0][1]*13}" r="3" fill="#d7f58a"/>' if pts else ''])
    parts.append('</svg>');return '\n'.join(parts)+'\n'

if __name__=='__main__':
    if len(sys.argv)!=2 or not os.path.isabs(sys.argv[1]):raise SystemExit('Explicit run: python receipt_recovery_008.py /absolute/output/dir')
    dest=sys.argv[1];os.makedirs(dest,exist_ok=True)
    e=evidence()
    with open(os.path.join(dest,'RECEIPT_RECOVERY_EVIDENCE.json'),'w') as f:json.dump(e,f,indent=2);f.write('\n')
    with open(os.path.join(dest,'RECEIPT_RECOVERY_PATHS.svg'),'w') as f:f.write(svg_gallery(e))
    print(json.dumps({'summary':e['summary'],'paired':e['paired_vs_receipt_rule'],'training':e['training'],'episodes':e['synthetic_episode_count'],'receipts':e['same_source_replayed_action_receipts'],'latency':e['inference_microseconds'],'time':e['total_wall_seconds'],'rss':e['process_maxrss_kib']},indent=2))
