"""C69 — preregistered synthetic-host memory intervention potency experiment.

Synthetic host only; original C54/58/60 interfaces are imported unchanged.
Select target and graph solely from pre-intervention host-observed route/menu
history, BEFORE executing or inspecting any policy suffix outcome.
"""
from __future__ import annotations
import argparse, base64, collections, copy, hashlib, json, math, os, sys, zlib
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
DATA_ROOT=Path(os.environ.get('C69_ARCHIVE_ROOT',str(HERE))).resolve()
sys.path.insert(0,str(DATA_ROOT/'frozen/research-source'))
import c60_remote_frontier_rl as c60
from experiment import sha as host_sha

SEED=690118
TICKS=(19,29)
DROPS=(0.,.5,1.)
ARMS=(('none',0.),('backtrack',.35),('backtrack',1.))
POLICIES=('C18-A0','C34-RESET','C60-FRONTIER','C60-FROZEN')
FAMILY_QUOTA=4
FAMILIES=('tree','braid','prim','wilson','kruskal')


def digest(b):return hashlib.sha256(b).hexdigest()

def known_ids():
    ids=set(json.loads((DATA_ROOT/'publication/C60-historical-graph-exclusions.json').read_text())['excluded_graphs'])
    a=json.loads((DATA_ROOT/'publication/C60-summary.json').read_text())
    for group in a['cohorts'].values():ids.update(r['graph'] for r in group)
    for loc,groups in [ ('results/C61-summary.json',('train_graphs','test_graphs')),
         ('lab/C64-summary.json',('graph_receipts',)),
         ('C65-summary.json',('graph_receipts',)),
         ('C66-summary.json',('graph_ids',)),
         ('C67-summary.json',('train_graph_ids','test_graph_ids')) ]:
        p=DATA_ROOT/loc
        if not p.exists():continue
        rep=json.loads(p.read_text())
        for key in groups:
            if key not in rep:continue
            def add(x):
                if isinstance(x,str):ids.add(x)
                elif isinstance(x,list):
                    for y in x:add(y)
                elif isinstance(x,dict):
                    if 'graph' in x:ids.add(x['graph'])
                    else:
                        for y in x.values():add(y)
            add(rep[key])
    return ids


def prepare_states(base):
    worldgen=c60.c59.c58.c55.c54.c52.worlds
    c52=c60.c59.c58.c55.c54.c52
    familiar=c52.c50.c47.c41.world
    alien=c52.c50.c47.c42.world
    seen=known_ids();excluded_before=len(seen)
    # Predeclared seed, families, ticks, route-richness threshold, quota and order.
    # No policy evaluation occurs before these selections.
    allstates=[];receipts=[];supply=collections.Counter();eligibility=[]
    for fi,family in enumerate(FAMILIES):
        fn=alien if family in ('wilson','kruskal') else familiar
        ws,rs=worldgen((family,),fn,SEED+fi*210001+311,10,exclude=seen)
        seen.update(r['graph'] for r in rs);receipts+=rs
        candidates=[]
        for w,info,seed,fam,case in ws:
            for tick in TICKS:
                pre=c60.c59.c58.c55.c54.prefix(base,w,seed,tick)
                if pre is None:
                    supply['prefix_terminal']+=1;continue
                snap,rng,log,receipt=pre
                # Target is observed frontier, not a hidden remote graph node.
                target=c60.c59.c58.c55.waypoint(log,snap.pos)
                if target is None:
                    supply['no_observed_route']+=1;continue
                route=log.route(snap.pos,target)
                if route is None or len(route)<4 or len(log.neighbors)<6 or len(log.actions)<19:
                    supply['too_little_memory']+=1;continue
                full=c60.c59.c58.damaged(log,info['graph_sha'],tick,1.)
                # Intentional potency guard: genuinely removed menu edges and
                # the original route must be unavailable after intervention.
                before=sum(map(len,log.neighbors.values()))
                after=sum(map(len,full.neighbors.values()))
                if before-after<3 or full.route(snap.pos,target) is not None:
                    supply['insufficient_route_erasure']+=1;continue
                candidates.append(((info['graph_sha'],tick), (snap,rng,log,receipt,tick,target,info,seed,fam,case,len(route)-1),
                    {'graph':info['graph_sha'],'family':family,'tick':tick,'route_edges':len(route)-1,
                     'prefix_menus':len(log.neighbors),'prefix_actions':len(log.actions),
                     'before_edges':before,'after_full_erasure_edges':after,
                     'route_absent_full_erasure':True,'prefix_sha':receipt['prefix_sha']}))
        # Stable original generation order: first eligible per graph; avoid
        # selecting multiple prefixes of one graph (cluster independence).
        graphset=set();chosen=[]
        for _,st,pr in candidates:
            if pr['graph'] in graphset:continue
            graphset.add(pr['graph']);chosen.append((st,pr))
            if len(chosen)==FAMILY_QUOTA:break
        eligibility.append({'family':family,'generated_worlds':len(ws),'eligible_prefixes':len(candidates),
                            'selected':len(chosen)})
        allstates+=chosen
    assert len(allstates)==len(FAMILIES)*FAMILY_QUOTA, ('insufficient predeclared cohort',eligibility,supply)
    ids=[r['graph'] for _,r in allstates]
    assert len(ids)==len(set(ids)) and not set(ids)&known_ids()
    return allstates,eligibility,receipts,dict(supply),excluded_before


def run_one(base,opt,st,policy,arm,drop,weights):
    # Instrument only the *local research ledger instance*. No historical
    # production interface/module implementation is changed or committed.
    orig=c60.c59.c58.damaged
    stats=collections.Counter();meta={}
    start=tuple(st[0].pos);goal=tuple(st[5]);original=st[2]
    def audit_damaged(log,graph,tick,fraction):
        led=orig(log,graph,tick,fraction)
        meta['menus_retained']=len(led.neighbors)
        meta['actions_retained']=len(led.actions)
        meta['route_retained']=int(led.route(start,goal) is not None)
        meta['edges_retained']=sum(len(n) for n in led.neighbors.values())
        route=led.route;frontiers=led.frontiers
        # Read telemetry: the policy actually interrogates route and frontier
        # memory at runtime; these calls are NOT evidence of route correctness.
        def audited_route(a,b):
            stats['route_reads']+=1
            r=route(a,b)
            if r is not None and len(r)>1:
                stats['successful_routes']+=1
                if any(tuple(x) in original.neighbors for x in r[:-1]):stats['prefix_route_read_hits']+=1
            return r
        def audited_frontiers():
            stats['frontier_reads']+=1
            fs=frontiers()
            if fs:stats['nonempty_frontiers']+=1
            return fs
        led.route=audited_route
        led.frontiers=audited_frontiers
        return led
    c60.c59.c58.damaged=audit_damaged
    try:
        policy_name='C60-ALIGNED' if policy=='C60-FROZEN' else policy
        z,_=c60.execute(base,opt,st,arm,policy_name,weights,drop,trace=True)
    finally:c60.c59.c58.damaged=orig
    assert z['host_rejections']==0 and z['protected_sha']==st[3]['protected_sha']
    return z,dict(stats)|meta


def path_metrics(z):
    ps=[tuple(z['start_pos'])]+[tuple(x['p']) for x in z['path']]
    travel=[(b[0]-a[0],b[1]-a[1]) for a,b in zip(ps,ps[1:]) if a!=b]
    turns=sum(a!=b for a,b in zip(travel,travel[1:]))
    rev=sum(a==(-b[0],-b[1]) for a,b in zip(travel,travel[1:]))
    n=max(1,len(travel));cnt=collections.Counter(travel)
    return {'turn_rate':turns/max(1,len(travel)-1), 'reverse_rate':rev/max(1,len(travel)-1),
            'unique_ratio':len(set(ps))/max(1,len(ps)),
            'direction_entropy':-sum(q/n*math.log2(q/n) for q in cnt.values()),
            'painting_fraction':sum(t['a']=='INK' for t in z['path'])/max(1,len(z['path']))}


def gallery(samples,p):
    cols=4;cw=240;ch=180;nrows=math.ceil(len(samples)/cols)
    x=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{cw*cols}" height="{ch*nrows}" viewBox="0 0 {cw*cols} {ch*nrows}">',
      '<rect width="100%" height="100%" fill="#111b26"/>']
    for i,r in enumerate(samples):
        ox=i%cols*cw;oy=i//cols*ch
        ps=[r['start']]+[a['p'] for a in r['path']]
        pts=ps+[r['goal_pos']]
        xx=[v[0] for v in pts]; yy=[v[1] for v in pts]
        mx=min(xx)-1;my=min(yy)-1;scale=min(211/max(1,max(xx)-mx+1),112/max(1,max(yy)-my+1))
        def pt(v):return f'{ox+9+(v[0]-mx)*scale:.1f},{oy+32+(v[1]-my)*scale:.1f}'
        col='#77d5f5' if r['goal'] else '#eeb57a'
        x.extend([f'<rect x="{ox+2}" y="{oy+2}" width="236" height="176" rx="6" fill="#1b2b3b"/>',
          f'<text x="{ox+9}" y="{oy+17}" fill="white" font-family="monospace" font-size="10">{r["family"]} {r["policy"]} drop={r["drop"]}</text>',
          f'<polyline points="{" ".join(pt(v) for v in ps)}" fill="none" stroke="{col}" stroke-width="1.6"/>',
          f'<circle cx="{pt(r["goal_pos"]).split(",")[0]}" cy="{pt(r["goal_pos"]).split(",")[1]}" r="3.5" fill="#e9d560"/>',
          f'<text x="{ox+9}" y="{oy+168}" fill="#aebfca" font-family="monospace" font-size="10">{len(r["path"])} actions | goal={r["goal"]}</text>'])
    x.append('</svg>');p.write_text('\n'.join(x)+'\n')


def run(output):
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    c60.HORIZON=95
    base,opt,_=c60.c59.c58.c55.c54.c52.c50.c47.c44.prerequisites(DATA_ROOT/'frozen/predecessor_C42')
    states,eligibility,receipts,skips,exclusions=prepare_states(base)
    prior=np.asarray(json.loads((DATA_ROOT/'publication/C60-summary.json').read_text())['weights']['C60-ALIGNED'],np.float32)
    rows=[];gallery_rows=[];actions_executed=0
    for i,(st,receipt) in enumerate(states):
        print(f'C69 graph {i+1}/{len(states)} {receipt["family"]} {receipt["tick"]}',flush=True)
        for arm in ARMS:
            for drop in DROPS:
                out={};telemetry={}
                for policy in POLICIES:
                    z,t=run_one(base,opt,st,policy,arm,drop,prior)
                    actions_executed+=z['steps'];telemetry[policy]=t
                    out[policy]={k:z[k] for k in ('goal','steps','new_sites','distance','action_sha','ink_sha','protected_sha','host_rejections')}|{'personality':path_metrics(z)}
                    if i<4 and drop in (0.,1.) and arm==ARMS[-1] and policy in ('C60-FRONTIER','C60-FROZEN'):
                        gallery_rows.append({'graph':receipt['graph'],'family':receipt['family'],'policy':policy,'drop':drop,
                          'goal':z['goal'],'goal_pos':z['target'],'start':z['start_pos'],'path':z['path'],
                          'action_sha':z['action_sha'],'ink_sha':z['ink_sha']})
                if arm[0]=='none':
                    assert len({(z['action_sha'],z['ink_sha']) for z in out.values()})==1
                rows.append({'graph':receipt['graph'],'family':receipt['family'],'tick':receipt['tick'],
                  'prefix_sha':receipt['prefix_sha'],'arm':list(arm),'drop':drop,'outcomes':out,'reads':telemetry})
    assert len(rows)==len(states)*len(ARMS)*len(DROPS)
    # Pair each graph+cue+policy at intact vs 50% and 100% erased memory.
    pairing=[]
    for _,r in states:
        for arm in ARMS:
            group={d:next(x for x in rows if x['graph']==r['graph'] and x['arm']==list(arm) and x['drop']==d) for d in DROPS}
            for p in POLICIES:
                ref=group[0.]['outcomes'][p]
                for d in DROPS[1:]:
                    tst=group[d]['outcomes'][p]
                    pairing.append({'graph':r['graph'],'family':r['family'],'arm':list(arm),'policy':p,'drop':d,
                     'action_changed':(ref['action_sha'],ref['ink_sha'])!=(tst['action_sha'],tst['ink_sha']),
                     'goal_delta':tst['goal']-ref['goal'],
                     'removed_menus':group[0.]['reads'][p]['menus_retained']-group[d]['reads'][p]['menus_retained'],
                     'removed_actions':group[0.]['reads'][p]['actions_retained']-group[d]['reads'][p]['actions_retained'],
                     'route_lost':group[0.]['reads'][p]['route_retained']-group[d]['reads'][p]['route_retained'],
                     'route_reads':group[d]['reads'][p].get('route_reads',0),
                     'prefix_route_read_hits':group[d]['reads'][p].get('prefix_route_read_hits',0)})
    agg={}
    for d in DROPS[1:]:
        for a in ARMS:
            for p in POLICIES:
                rr=[r for r in pairing if r['drop']==d and r['arm']==list(a) and r['policy']==p]
                agg[f'{d}/{a[1]}/{p}']={'pairs':len(rr),'changed_traces':sum(r['action_changed'] for r in rr),
                   'goal_deltas':sum(r['goal_delta'] for r in rr),'goals_won':sum(r['goal_delta']>0 for r in rr),
                   'goals_lost':sum(r['goal_delta']<0 for r in rr),'lost_route':sum(r['route_lost']>0 for r in rr),
                   'removed_menus_total':sum(r['removed_menus'] for r in rr),
                   'route_reads_total':sum(r['route_reads'] for r in rr),
                   'prefix_route_read_hits':sum(r['prefix_route_read_hits'] for r in rr)}
    summary={'schema':'driv3r-c69-memory-potency/1','seed':SEED,'ticks':TICKS,'family_quota':FAMILY_QUOTA,
       'families':FAMILIES,'predeclared_thresholds':{'menus':6,'prefix_actions':19,'route_edges':3,
         'erased_edges':3,'full_erasure_route_absent':True},
       'eligible':eligibility,'eligibility_skips':skips,'known_graph_exclusions':exclusions,
       'selected_prefixes':[r for _,r in states],'generated_graph_receipts':receipts,
       'conditions':len(rows),'host_executed_suffixes':len(rows)*len(POLICIES),
       'host_actions_executed':actions_executed,'paired':agg,
       'gallery_traces':len(gallery_rows),'fixed_frozen_policy':'C60-ALIGNED',
       'limitations':['Preselection by pre-intervention observed route richness, not representative of remote unseen targets',
         'No new trained neural weights; C68 explicitly required manipulation qualification before training',
         'No native CyberSand runtime qualification; the host is an immutable synthetic research fixture',
         'Memory read telemetry is only observed ledger route/frontier calls and does not establish causal attribution by itself',
         'Same RNG stream across paired arms; divergence may propagate after one changed action',
         'Goal comparison exploratory and small; outcomes were not used in eligibility filtering']}
    ledger={'schema':'driv3r-c69-executed-ledger/1','summary':summary,'rows':rows,'pairs':pairing,'gallery':gallery_rows}
    (output/'C69-summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (output/'C69-ledger.zlib.b64').write_text(base64.b64encode(zlib.compress(json.dumps(ledger,sort_keys=True,separators=(',',':')).encode(),9)).decode()+'\n')
    gallery(gallery_rows,output/'C69-gallery.svg')
    print(json.dumps({'selection':eligibility,'suffixes':summary['host_executed_suffixes'],'actions':actions_executed,
      'no_guidance_full':{p:agg[f'1.0/0.0/{p}'] for p in POLICIES},
      'strong_full':{p:agg[f'1.0/1.0/{p}'] for p in POLICIES}},indent=2),flush=True)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out');args=parser.parse_args();run(args.out)
