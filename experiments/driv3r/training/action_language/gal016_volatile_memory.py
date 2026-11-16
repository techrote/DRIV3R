"""GAL-016: bounded material memory and host-owned transient changes.

External synthetic experiment only. The action menu, lease, mutations and
protected cells are owned by the synthetic host; policy receives radius-four
material observations and accepted action receipts, never hidden topology.
A checkpoint from GAL-015 is reused, not retrained; model-call counts are
proxy counts and not resident SLM measurements.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import statistics
from frontier_memory import Ledger
from lab import START, DECISION_CAP, sha
from material_corridors import visible_material, digest
from route_scaffold import HorizonWorld, PaintScorer, load_frozen, choose

SCHEMA='gal-016-volatile-memory/1'
EVAL_BASE=2640000
KINDS=('path','tree','comb','deep_tree')
TARGETS=(48,64)
CAPS=(81,161,321,None)
DYNAMICS=('static','pulse')
EVENT_TURNS=(8,14,26,32)

class BoundedLedger(Ledger):
    """Finite spatial evidence, with receipt-derived paint memory held to cap.

    Keys are refreshed only by sensor data or accepted receipts. Historical
    occupancy is discarded at capacity, not inferred from hidden target state.
    """
    def __init__(self,capacity=None):
        if capacity is not None and (capacity<41 or capacity>4096):
            raise ValueError('invalid spatial memory capacity')
        super().__init__();self.capacity=capacity;self.known=OrderedDict()
        self.evicted=0

    def _remember(self,p,m):
        if p in self.known:self.known.move_to_end(p)
        self.known[p]=m
        if self.capacity is not None:
            while len(self.known)>self.capacity:
                q,_=self.known.popitem(last=False)
                self.recorded_paints.discard(q)
                self.visited.pop(q,None)
                self.evicted+=1

    def observe(self,w):
        crop=visible_material(w,'material')
        for p,m in sorted(crop.items()):
            self._remember(p,m)
            # Material is authoritative: transient host changes can invalidate
            # a formerly recorded paint. Never preserve contradictory ink.
            if m!=58:self.recorded_paints.discard(p)
            elif m==58:self.recorded_paints.add(p)
        self.observations+=1
        return crop

    def accept(self,action,path,receipt):
        assert receipt['path']==path and receipt['action_id']==action
        paint=action.startswith(('INK_','JOG_'))
        for p in path:
            self.visited[p]=self.visited.get(p,0)+1
            if paint:
                self._remember(p,58)
                self.recorded_paints.add(p)
        if self.capacity is not None:
            keep=set(self.known)
            self.visited={p:n for p,n in self.visited.items() if p in keep or p==START}
            self.recorded_paints.intersection_update(keep)

class PulseWorld(HorizonWorld):
    """External host inserts/removes one inert, nonprotected, unpainted block.

    Host interventions are separate from policy receipts and occur *after*
    accepted actions. Policy sees interventions only through later crops and
    host-authorized legal catalogs. Mutations never touch protected or ink.
    """
    def __init__(self,seed,kind,arm,lease,target,dynamic):
        super().__init__(seed,kind,arm,lease,target)
        if dynamic not in DYNAMICS:raise ValueError('unknown host dynamics')
        self.dynamic=dynamic;self.events=[];self.block=None

    def _change(self,at,value,label):
        before=sha(self.material);old=self.material[self.idx(at)]
        assert at not in self.protected and at!=self.cursor and at not in self.own
        assert old in (0,3) and value in (0,3) and old!=value
        self.material[self.idx(at)]=value
        self.events.append({'at_decision':self.decisions,'point':at,'kind':label,
             'from':old,'to':value,'before_material_sha256':before,
             'after_material_sha256':sha(self.material)})

    def _advance(self):
        if self.dynamic!='pulse':return
        if self.decisions in (EVENT_TURNS[0],EVENT_TURNS[2]):
            # Host-only deterministic choice. Prefer a far unpainted target
            # that is not the cursor, minimizing degenerate immediate blocks.
            candidates=[p for p in self.goal if self.material[self.idx(p)]==0 and p!=self.cursor]
            if candidates:
                chosen=max(candidates,key=lambda p:(abs(p[0]-self.cursor[0])+abs(p[1]-self.cursor[1]),p))
                self._change(chosen,3,'obstruct');self.block=chosen
        elif self.decisions in (EVENT_TURNS[1],EVENT_TURNS[3]) and self.block is not None:
            self._change(self.block,0,'release');self.block=None

    def apply(self,name,token,*,original_response=None):
        r=super().apply(name,token,original_response=original_response)
        self._advance()
        return r

def episode(seed,kind,target,capacity,dynamic,base,adapter,trace=False):
    lease=f'gal016:{kind}:{target}:{seed}:{capacity}:{dynamic}'
    w=PulseWorld(seed,kind,'lift',lease,target,dynamic)
    led=BoundedLedger(capacity)
    calls=0;route_steps=0
    import random
    rng=random.Random(f'gal016:choice:{kind}:{target}:{seed}')
    while not w.stopped and w.decisions<DECISION_CAP and w.goal-w.own:
        led.observe(w)
        action,token,call,reason=choose(w,led,base,adapter,'route_trained',rng)
        offered,_=w.offer()
        receipt=w.apply(action,token,original_response=f'gal016:{reason}:{action}')
        led.accept(action,offered[action],receipt)
        calls+=call;route_steps+=reason=='route'
    assert not (w.own&w.protected)
    assert all(w.material[w.idx(p)]==w.original[w.idx(p)] for p in w.protected)
    assert len(led.known)<=capacity if capacity is not None else True
    complete=w.goal.issubset(w.own)
    row={'seed':seed,'kind':kind,'target':target,'capacity':capacity,'dynamic':dynamic,
        'complete':complete,'correct':len(w.own&w.goal),
        'coverage':round(len(w.own&w.goal)/target,6),
        'decisions':w.decisions,'model_calls_proxy':calls,'route_steps':route_steps,
        'known_cells':len(led.known),'evictions':led.evicted,
        'events':len(w.events),'unreleased_obstruction':int(w.block is not None),
        'stopped_early':w.stopped and not complete,
        'material_sha256':sha(w.material),'receipts_sha256':sha(w.trace),
        'events_sha256':sha(w.events),'native_receipts':0,'resident_slms':0}
    return (w,led,row) if trace else (None,None,row)

def aggregate(rows):
    return {'n':len(rows),'complete':sum(r['complete'] for r in rows),
        'mean_coverage':round(statistics.mean(r['coverage'] for r in rows),5),
        'mean_decisions':round(statistics.mean(r['decisions'] for r in rows),4),
        'mean_model_calls_proxy':round(statistics.mean(r['model_calls_proxy'] for r in rows),4),
        'mean_evictions':round(statistics.mean(r['evictions'] for r in rows),2),
        'with_obstruction':sum(bool(r['events']) for r in rows),
        'early_stops':sum(r['stopped_early'] for r in rows)}

def audit_trace(w,row):
    """Independent host replay: reconstruct material interventions and leases."""
    other=PulseWorld(row['seed'],row['kind'],'lift',w.lease,row['target'],row['dynamic'])
    for receipt in w.trace:
        menu,lease=other.offer()
        assert lease==receipt['offered_sha256']
        assert menu[receipt['action_id']]==receipt['path']
        seen=other.apply(receipt['action_id'],lease,original_response=receipt['original_response'])
        assert seen==receipt
    assert other.events==w.events
    assert sha(other.trace)==row['receipts_sha256']
    assert sha(other.material)==row['material_sha256']
    assert sha(other.events)==row['events_sha256']
    return True

def run(out,frozen,paint,eval_per_panel=8):
    if not 2<=eval_per_panel<=24:raise ValueError('bounded evaluation required')
    base=load_frozen(frozen)
    cp=json.loads(Path(paint).read_text())
    if cp.get('schema')!='gal-015-route-scaffold/1' or len(cp['weights'])!=6:
        raise ValueError('incompatible trained paint scorer')
    adapter=PaintScorer(cp['weights'])
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[];trace_gallery={};audit_count=0
    for kind in KINDS:
        for target in TARGETS:
            for i in range(eval_per_panel):
                seed=EVAL_BASE+i
                for dynamic in DYNAMICS:
                    for cap in CAPS:
                        keep=(i==0 and kind in ('tree','deep_tree') and target==64)
                        w,led,row=episode(seed,kind,target,cap,dynamic,base,adapter,trace=keep)
                        rows.append(row)
                        if keep:
                            assert audit_trace(w,row);audit_count+=1
                            k=f'{kind}:{dynamic}:{cap}'
                            trace_gallery[k]={'row':row,'material':w.material,
                                'corridor':sorted(w.corridor),'events':w.events,'receipts':w.trace}
    panels={f'{d}:{str(c)}':aggregate([r for r in rows if r['dynamic']==d and r['capacity']==c])
            for d in DYNAMICS for c in CAPS}
    totals={d:aggregate([r for r in rows if r['dynamic']==d]) for d in DYNAMICS}
    results={'schema':SCHEMA,'eval_seed_range':[EVAL_BASE,EVAL_BASE+eval_per_panel-1],
       'eval_per_panel':eval_per_panel,'topologies':KINDS,'targets':TARGETS,
       'spatial_caps':CAPS,'dynamics':DYNAMICS,'panels':panels,'totals':totals,
       'episodes':len(rows),'independently_host_replayed_gallery_episodes':audit_count,
       'frozen_memory_checkpoint_sha256':digest(frozen),
       'trained_paint_checkpoint_sha256':digest(paint),
       'real_optimizer_updates_this_experiment':0,'native_receipts':0,
       'resident_slms':0,'model_calls_are_only_proxies':True,
       'external_synthetic_only':True}
    for name,payload in [('GAL016_metrics.json',results),('GAL016_episodes.json',rows),
                         ('GAL016_receipt_gallery.json',trace_gallery)]:
        (out/name).write_text(json.dumps(payload,sort_keys=True,indent=2)+'\n')
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--frozen',required=True)
    p.add_argument('--paint',required=True);p.add_argument('--eval',type=int,default=8)
    a=p.parse_args();print(json.dumps(run(a.out,a.frozen,a.paint,a.eval),indent=2))
