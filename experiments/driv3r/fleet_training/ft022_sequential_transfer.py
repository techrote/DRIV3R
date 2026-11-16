"""FT-022: held-out sequential legal-world transfer of FT-021 policies.

Explicit, bounded synthetic experiment. No CyberSand runtime/ABI claims. Four
independently trained 20x24 policies, optional legal-route and recovery updates.
Source/checkpoint-hash custody, full legality, protected-cell, and trace replay.
"""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time
from zipfile import ZipFile
import numpy as np

ID = 'FT-022-20261116-sequential-legal-transfer'
SCHEMA = 'driv3r.fleet_training.ft022/1'
NICHES = ('navigator','recoverer','branch_artist','material_cautious')
PARENT_SHA = {
 'navigator':'6f08bbec7f5fb51b7055803b2d57226bf87743d03297b35f59d95703c6308dff',
 'recoverer':'ac542e34ba48025555633a950ac48fdd1f099517f0eedc295e46cfb1e2915390',
 'branch_artist':'3f1a560423a1f77a4bdcce17604e4a13eb5e410cbbaf535fb913de5a4ed9ee73',
 'material_cautious':'596b9482b27c3dccf542ad40a599d2e8fbf320ddb415604f1bb04f2dd3edb770',
}
FEATURES=('bias','paint','retrace','stop','goal_progress','onward','blocked','junction',
 'turn','frontier','material_edge','paint_density','early','protected_near','goal_switched',
 'direction_x','direction_y','guidance','motif','recovery_need')
DIRS=(('N',(0,-1)),('E',(1,0)),('S',(0,1)),('W',(-1,0)))
DELTA=dict(DIRS)
ARMS=('frozen','route','route_recovery')
MAX_STEPS=100
UPDATES=360
WORLD_TRAIN=16
WORLD_TEST=12
SIDE=16

def seed(*parts):
 return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'big') % 2**32

def digest(data):
 return hashlib.sha256(data).hexdigest()

def filehash(p):return digest(Path(p).read_bytes())

def canon(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'))

def softmax(z):
 a=np.exp(np.clip(z-np.max(z),-40,0));return a/a.sum()

class Policy:
 def __init__(self,parameters):
  self.p={k:np.asarray(v,dtype=np.float64) for k,v in parameters.items()}
  if {k:v.shape for k,v in self.p.items()}!={'W1':(20,24),'b1':(24,),'W2':(24,),'b2':(1,)}:
   raise ValueError('model shape mismatch')
  self.m={k:np.zeros_like(v) for k,v in self.p.items()}
  self.v={k:np.zeros_like(v) for k,v in self.p.items()}
  self.updates=0
 def logits(self,X):return np.tanh(X@self.p['W1']+self.p['b1'])@self.p['W2']+self.p['b2'][0]
 def teach(self,X,y):
  H=np.tanh(X@self.p['W1']+self.p['b1']); pr=softmax(H@self.p['W2']+self.p['b2'][0]);d=pr-y
  dh=d[:,None]*self.p['W2'][None,:]*(1-H*H)
  g=dict(W1=X.T@dh,b1=dh.sum(0),W2=H.T@d,b2=np.array([d.sum()]))
  self.updates+=1
  for k,v in g.items():
   self.m[k]=.9*self.m[k]+.1*v;self.v[k]=.999*self.v[k]+.001*v*v
   self.p[k]-=.005*(self.m[k]/(1-.9**self.updates))/(np.sqrt(self.v[k]/(1-.999**self.updates))+1e-8)
  return -float(np.dot(y,np.log(np.maximum(pr,1e-12))))
 def record(self,niche,arm,parent):
  return dict(schema=SCHEMA,niche=niche,arm=arm,parent_sha256=parent,adam_updates=self.updates,
    feature_names=FEATURES,parameters={k:v.tolist() for k,v in self.p.items()})

def parents(z):
 models={}
 for niche in NICHES:
  raw=z.read('ft021_long/one_resident/'+niche+'.json')
  if digest(raw)!=PARENT_SHA[niche]:raise ValueError('Original FT021 parent hash mismatch: '+niche)
  obj=json.loads(raw)
  if obj.get('schema')!='driv3r.fleet_training.ft021/1' or obj.get('niche')!=niche or obj.get('new_adam_updates')!=900:
   raise ValueError('Parent provenance mismatch: '+niche)
  models[niche]=obj['parameters']
 return models

def inside(p):return 0<=p[0]<SIDE and 0<=p[1]<SIDE

def adj(p):
 for name,(dx,dy) in DIRS:
  q=(p[0]+dx,p[1]+dy)
  if inside(q):yield name,q

def route(pos,goal,occupied):
 q=deque([pos]);prev={pos:None}
 while q:
  p=q.popleft()
  if p==goal:break
  for name,n in adj(p):
   if n not in occupied and n not in prev:prev[n]=(p,name);q.append(n)
 if goal not in prev:return []
 out=[];p=goal
 while p!=pos:
  p,name=prev[p];out.append(name)
 return out[::-1]

class World:
 def __init__(self,domain,world_id):
  if domain not in ('train','open','guarded'):raise ValueError('invalid world domain')
  self.domain=domain;self.world_id=world_id; rng=np.random.default_rng(seed(ID,domain,world_id))
  self.protected={(x,y) for x in range(6,9) for y in range(5,10)}
  if domain=='guarded':self.protected |= {(x,12) for x in range(1,10)}
  prob={'train':.11,'open':.15,'guarded':.22}[domain]
  self.obstacles={(x,y) for x in range(SIDE) for y in range(SIDE) if rng.random()<prob} - self.protected
  occupied=self.protected|self.obstacles
  free=[(x,y) for x in range(SIDE) for y in range(SIDE) if (x,y) not in occupied]
  def reach(src):
   todo=deque([src]);dist={src:0}
   while todo:
    p=todo.popleft()
    for _,n in adj(p):
     if n not in occupied and n not in dist:
      dist[n]=dist[p]+1;todo.append(n)
   return dist
  for attempt in range(32):
   self.start=free[int(rng.integers(len(free)))]
   distances=reach(self.start)
   targets=[p for p,d in distances.items() if 6<=d<=22]
   if not targets:continue
   self.goal1=targets[int(rng.integers(len(targets)))]
   d2=reach(self.goal1)
   targets2=[p for p,d in d2.items() if p!=self.goal1 and 6<=d<=22]
   if not targets2:continue
   self.goal2=targets2[int(rng.integers(len(targets2)))];break
  else:raise ValueError('World failed reachable two-goal fixture')
  self.cursor=self.start;self.trail=[self.start];self.painted={self.start}
  self.steps=0;self.switched=False;self.stopped=False;self.ending=None;self.heading=None
  self.visited={self.start}
  self.material={p:int((p[0]*7+p[1]*11+(p[0]//3)*5)%3) for p in free}
 def goal(self):return self.goal2 if self.switched else self.goal1
 def legal(self):
  if self.stopped:return []
  moves=[n for n,q in adj(self.cursor) if q not in (self.obstacles|self.protected)]
  if len(self.trail)>=2 and self.trail[-2] in [q for _,q in adj(self.cursor)]:moves.append('RETRACE')
  return moves+['STOP']
 def after(self,a):
  if a=='RETRACE':return self.trail[-2]
  if a=='STOP':return self.cursor
  dx,dy=DELTA[a];return (self.cursor[0]+dx,self.cursor[1]+dy)
 def features(self,actions):
  rows=[]
  dist=lambda p:abs(p[0]-self.goal()[0])+abs(p[1]-self.goal()[1])
  for a in actions:
   n=self.after(a); neigh=list(adj(n)); open_n=[p for _,p in neigh if p not in (self.obstacles|self.protected)]
   nearp=sum(p in self.protected for _,p in neigh)
   dx=n[0]-self.cursor[0];dy=n[1]-self.cursor[1]
   progress=(dist(self.cursor)-dist(n))/2.
   matdiff=int(self.material.get(self.cursor,0)!=self.material.get(n,0))
   unvisited=sum(p not in self.visited for p in open_n)
   rows.append([1.,float(a in DELTA and n not in self.painted),float(a=='RETRACE'),float(a=='STOP'),
    progress,len(open_n)/4.,(len(neigh)-len(open_n))/4.,float(len(open_n)>=3),
    float(self.heading is not None and self.heading!=a),unvisited/4.,float(matdiff),
    (len(self.painted & set(open_n)))/4.,self.steps/MAX_STEPS,nearp/4.,float(self.switched),
    float(dx),float(dy),float(progress>0),float((n[0]+2*n[1])%5==0),
    float(a=='RETRACE' and (unvisited==0 or dist(n)<dist(self.cursor)))])
  return np.array(rows,dtype=np.float64)
 def state(self):
  return digest(canon([self.domain,self.world_id,self.cursor,self.trail,sorted(self.painted),self.steps,self.switched,self.stopped,self.ending]).encode())
 def step(self,a):
  menu=self.legal()
  if a not in menu:raise ValueError('illegal action: '+str(a))
  if a=='STOP':
   self.ending=('final_goal' if self.switched and self.cursor==self.goal2 else 'host_forced' if len(menu)==1 else 'premature');self.stopped=True
  elif a=='RETRACE':
   self.cursor=self.trail[-2];self.trail.pop();self.heading=a
  else:
   n=self.after(a)
   if n in self.trail:self.trail=self.trail[:self.trail.index(n)+1]
   else:self.trail.append(n)
   self.cursor=n;self.heading=a;self.painted.add(n);self.visited.add(n)
  self.steps+=1
  if not self.switched and self.cursor==self.goal1:
   self.switched=True
  if self.painted&(self.protected|self.obstacles):raise AssertionError('protected or blocked cell painted')
  if self.steps>=MAX_STEPS and not self.stopped:self.stopped=True;self.ending='horizon'

def teacher(w):
 if w.switched and w.cursor==w.goal2:return 'STOP'
 path=route(w.cursor,w.goal(),w.protected|w.obstacles)
 if not path:return None
 a=path[0]
 if len(w.trail)>=2 and w.after(a)==w.trail[-2]:return 'RETRACE'
 return a

def sample(w,a):
 menu=[m for m in w.legal() if m!='STOP']
 if a not in menu or len(menu)<2:return None
 X=w.features(menu);y=np.zeros(len(menu));y[menu.index(a)]=1
 return X,y

def training_set(niche,domain,count,recovery):
 examples=[];sources=[];unique=set()
 for i in range(count):
  wid=seed(ID,niche,domain,i)
  if wid in unique:raise AssertionError('duplicate world lease')
  unique.add(wid);w=World('train',wid)
  for step in range(MAX_STEPS):
   if w.stopped:break
   a=teacher(w)
   if a is None:break
   if a=='STOP':w.step(a);break
   e=sample(w,a)
   if e:examples.append(e)
   # Occasionally inject a valid non-goal detour, then teach lawful recovery.
   if recovery and step%11==4:
    cand=[x for x in w.legal() if x not in ('STOP',a)]
    if cand:
     detour=cand[seed(ID,niche,domain,i,step)%len(cand)]
     w.step(detour)
     next_action=teacher(w)
     if next_action and next_action!='STOP':
      re=sample(w,next_action)
      if re:examples.append(re)
     continue
   w.step(a)
  sources.append(dict(world_id=wid,demonstration_steps=w.steps,goal_switched=w.switched,ending=w.ending))
 return examples,sources

def train(par,niche,arm):
 p=Policy(par)
 if arm=='frozen':return p,[]
 examples,_=training_set(niche,'train',WORLD_TRAIN,arm=='route_recovery')
 if len(examples)<80:raise ValueError('insufficient legal teacher examples')
 rng=np.random.default_rng(seed(ID,niche,arm,'optimizer'));loss=[]
 for k in range(UPDATES):
  X,y=examples[int(rng.integers(len(examples)))];loss.append(p.teach(X,y))
 return p,loss

def rollout(params,domain,wid,trace=False):
 w=World(domain,wid);p=Policy(params);receipts=[];retrace_count=0
 while not w.stopped:
  menu=w.legal()
  if not menu:raise AssertionError('unexpected empty action menu')
  if w.switched and w.cursor==w.goal2:a='STOP'
  else:
   moves=[x for x in menu if x!='STOP']
   a=moves[int(np.argmax(p.logits(w.features(moves))))] if moves else 'STOP'
  before=w.state();w.step(a);retrace_count+=int(a=='RETRACE')
  if trace:receipts.append(dict(turn=w.steps,before=before,menu=menu,choice=a,after=w.state()))
 result=dict(world_id=wid,domain=domain,ending=w.ending,steps=w.steps,
  completed=int(w.ending=='final_goal'),goal_switched=int(w.switched),
  painted=len(w.painted),retrace_actions=retrace_count,
  protected_edited=len(w.painted&(w.protected|w.obstacles)),terminal_cursor=w.cursor)
 if trace:
  clone=World(domain,wid)
  for r in receipts:
   if clone.state()!=r['before'] or clone.legal()!=r['menu']:raise AssertionError('pre-state/legality replay drift')
   clone.step(r['choice'])
   if clone.state()!=r['after']:raise AssertionError('post-state replay drift')
  if clone.state()!=w.state():raise AssertionError('replay terminal mismatch')
  result['receipts']=receipts
 return result,w

def svg(w):
 scale=14;colors=['#5dc6a9','#5a8ddb','#f3b45c']
 elements=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIDE*scale} {SIDE*scale}"><rect width="100%" height="100%" fill="#15202b"/>']
 for x in range(SIDE):
  for y in range(SIDE):
   p=x,y
   c=('#354b58' if p in w.obstacles else '#a44d5c' if p in w.protected else colors[w.material[p]] if p in w.painted else '#23313f')
   elements.append(f'<rect x="{x*scale+1}" y="{y*scale+1}" width="{scale-2}" height="{scale-2}" fill="{c}"/>')
 for point,c in [(w.start,'#fff'),(w.goal1,'#f6d365'),(w.goal2,'#ee80c2'),(w.cursor,'#fff')]:
  elements.append(f'<circle cx="{point[0]*scale+scale/2}" cy="{point[1]*scale+scale/2}" r="3" fill="{c}"/>')
 return ''.join(elements)+'</svg>\n'

def run(zip_path,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 start=time.monotonic()
 r=dict(schema=SCHEMA,id=ID,status='starting',source_sha256=filehash(__file__),
  parent_zip_sha256=filehash(zip_path),parent_checkpoint_sha256=PARENT_SHA,
  budget=dict(max_steps=MAX_STEPS,optimizer_updates_per_child=UPDATES,
              training_worlds_per_niche_per_arm=WORLD_TRAIN,heldout_worlds_per_niche_per_domain=WORLD_TEST),
  models=[],groups=[],failures=[])
 try:
  with ZipFile(zip_path) as z:parent=parents(z)
  seen=set()
  for niche in NICHES:
   for arm in ARMS:
    p,loss=train(parent[niche],niche,arm)
    if arm!='frozen':
     d=out/'models';d.mkdir(exist_ok=True)
     f=d/(niche+'-'+arm+'.json');f.write_text(canon(p.record(niche,arm,PARENT_SHA[niche]))+'\n')
     r['models'].append(dict(lease=f'{ID}/{niche}/{arm}',sha256=filehash(f),updates=p.updates,
        first_loss=round(float(np.mean(loss[:36])),6),last_loss=round(float(np.mean(loss[-36:])),6)))
    for domain in ('open','guarded'):
     episodes=[]
     for i in range(WORLD_TEST):
      wid=seed(ID,niche,domain,i)
      key=(niche,domain,wid)
      if arm=='frozen':
       if key in seen:raise AssertionError('duplicate heldout dispatch')
       seen.add(key)
      elif key not in seen:raise AssertionError('unmatched world dispatch')
      e,w=rollout(p.p,domain,wid,trace=i==0)
      if i==0:
       d=out/'traces'/niche/domain;d.mkdir(parents=True,exist_ok=True)
       (d/(arm+'.json')).write_text(canon(e)+'\n')
       (d/(arm+'.svg')).write_text(svg(w))
      episodes.append({k:v for k,v in e.items() if k!='receipts'})
     r['groups'].append(dict(niche=niche,arm=arm,domain=domain,
       successes=sum(e['completed'] for e in episodes),goal_switches=sum(e['goal_switched'] for e in episodes),
       mean_painted=round(sum(e['painted'] for e in episodes)/len(episodes),4),
       mean_steps=round(sum(e['steps'] for e in episodes)/len(episodes),4),
       mean_retraces=round(sum(e['retrace_actions'] for e in episodes)/len(episodes),4),
       endings={k:sum(e['ending']==k for e in episodes) for k in ('final_goal','horizon','premature','host_forced')},
       episodes=episodes))
    if time.monotonic()-start>110:raise TimeoutError('FT022 bounded runtime exceeded')
  r['summary']={arm:{'completions':sum(g['successes'] for g in r['groups'] if g['arm']==arm),
          'goal_switches':sum(g['goal_switches'] for g in r['groups'] if g['arm']==arm),
          'mean_ink':round(sum(g['mean_painted'] for g in r['groups'] if g['arm']==arm)/8,4),
          'mean_retraces':round(sum(g['mean_retraces'] for g in r['groups'] if g['arm']==arm)/8,4)} for arm in ARMS}
  r['summary']['matched_worlds_per_arm']=len(seen)
  r['summary']['new_adam_updates']=sum(x['updates'] for x in r['models'])
  r['status']='completed'
 except BaseException as e:
  r['status']='failed';r['failures'].append(type(e).__name__+': '+str(e));raise
 finally:
  r['elapsed_seconds']=round(time.monotonic()-start,4)
  (out/'report.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
  (out/'SHA256SUMS').write_text(''.join(filehash(f)+'  '+str(f.relative_to(out))+'\n' for f in sorted(out.rglob('*')) if f.is_file() and f.name!='SHA256SUMS'))
 return r

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--parents',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
 a=ap.parse_args();r=run(a.parents,a.output)
 print(canon(dict(status=r['status'],summary=r['summary'],elapsed_seconds=r['elapsed_seconds'])))
