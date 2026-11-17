"""POST-SWITCH-REGRET-001: shared-prefix causal first-action audit.

Only DEVELOPMENT synthetic worlds and previously trained/checkpointed tiny models.
No training, no native CyberSand execution, no structural HELDOUT access.
Every first action comes from the host-legal menu; all subsequent actions are
selected autonomously by one of three frozen checkpointed suffix policies.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, resource, statistics, time
from collections import Counter
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from . import goal_switch_curriculum_lab as prior, goal_recovery_lab as base, policy
from .world import World, Spec, MOVE, digest
from .train import sha_file

VERSION='post-switch-regret/1'
SEEDS=(619,620)
ARMS=prior.ARMS
LOOKAHEAD=16
DEPENDENCIES=('post_switch_regret_lab.py','goal_switch_curriculum_lab.py','goal_recovery_lab.py',
              'goal_retarget_lab.py','goal_conditioning_lab.py','policy.py','world.py','train.py')

def sources():
    return {x:sha_file(Path(__file__).parent/x) for x in DEPENDENCIES}

def load_models(checkpoints, seeds=SEEDS):
    models={}; hashes={}
    for seed in seeds:
        models[seed]={}; hashes[str(seed)]={}
        for arm in ARMS:
            p=checkpoints/f'{seed}-{arm}.npz'
            m=base.Ranker.load(p)
            if m.updates<1 or len(m.p['w1'])!=base.WIDTH: raise ValueError('checkpoint shape')
            models[seed][arm]=m; hashes[str(seed)][arm]=sha_file(p)
    return models,hashes

def distance(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])

def vote(model,w,target):
    menu,x=base.matrix(w,target,160,False)
    p=model.probs(x,len(menu))
    if menu!=w.menu(12) or len(menu)>policy.ACTION_MAX or abs(float(sum(p))-1)>1.e-8:
        raise ValueError('host menu or policy distribution drift')
    return {'legal':menu,'probs':[round(float(v),10) for v in p], 'chosen':menu[int(p.argmax())]}

def prefix_states(sp,seed,variant,model):
    if sp.split!='development':raise ValueError('cannot open heldout')
    w=World.create(sp);goals=prior.goals_for(w,seed,variant)
    states=[]; prior_rows=[]
    for t in range(160):
        if w.stopped:break
        if t in prior.SWITCHES:
            p=sum(t>=s for s in prior.SWITCHES)
            states.append({'step':t,'phase':p,'goal':goals[p],'world':copy.deepcopy(w),
                           'prefix_hash':digest(w.decisions), 'base_hash':w.base_sha(),
                           'previous_actions':list(prior_rows)})
        phase=sum(t>=s for s in prior.SWITCHES)
        q=vote(model,w,goals[phase]); prior_rows.append(w.apply(q['chosen'],12))
    return states,{'decisions':len(prior_rows),'last_state_sha':w.state_sha(),'stopped_early':w.stopped and w.steps<160,
                   'missing_switches':sorted(set(prior.SWITCHES)-{x['step'] for x in states}),
                   'prefix_transcript_sha':digest(prior_rows)}

def suffix(w,first,target,model):
    start_cursor=w.cursor; start_dist=distance(start_cursor,target);pre_recovery=w.recoveries
    first_menu=w.menu(12)
    if first not in first_menu:raise ValueError('nonlegal fork')
    rows=[];hit=None
    for i in range(LOOKAHEAD):
        if w.stopped:break
        q=vote(model,w,target)
        action=first if i==0 else q['chosen']
        ev=w.apply(action,12)
        if w.cursor==target and hit is None:hit=i+1
        rows.append({'menu':q['legal'],'probs':q['probs'], 'model_chosen':q['chosen'],
                     'executed':action, 'receipt':ev})
    return {'first':first,'start_distance':start_dist,'final_distance':distance(w.cursor,target),
            'min_distance':min([start_dist]+[distance(x['receipt']['cursor_after'],target) for x in rows]),
            'hit_step':hit,'added_recoveries':w.recoveries-pre_recovery,
            'added_ink':sum(bool(x['receipt']['painted']) for x in rows),
            'stopped':w.stopped,'end_state_sha':w.state_sha(),'trace_sha':digest(w.decisions), 'rows':rows}

def best_action(outcomes):
    """Predeclared lexicographic goal priority, explicit ties allowed.
    Hit, earliest hit; otherwise final distance; then recovery; then STOP.
    """
    def key(v):
        return (int(v['hit_step'] is not None),
                -(v['hit_step'] or 0) if v['hit_step'] is not None else -v['final_distance'],
                -v['added_recoveries'],int(not v['stopped']))
    best=max(key(v) for v in outcomes.values())
    return sorted(a for a,v in outcomes.items() if key(v)==best)

def evaluate_state(state,models,seed,sp,variant,retain=True):
    w=state['world']; target=state['goal'];menu=w.menu(12)
    if not menu:raise ValueError('state has no host menu')
    votes={arm:vote(models[seed][arm],w,target) for arm in ARMS}
    results={}; best={}
    for suffix_arm in ARMS:
        rr={}
        for a in menu:rr[a]=suffix(copy.deepcopy(w),a,target,models[seed][suffix_arm])
        results[suffix_arm]=rr;best[suffix_arm]=best_action(rr)
    rec={'seed':seed,'spec':sp.__dict__,'variant':variant,'phase':state['phase'],
      'switch_step':state['step'],'target':list(target),'goal_already_ink':target in w.ink,
      'world_start_sha':state['base_hash'],'prefix_hash':state['prefix_hash'],
      'prefix_actions':len(state['previous_actions']),'prefix_steps':w.steps,
      'recovery_at_switch':w.recoveries,'logical_moves_at_switch':w.logical_moves,
      'path_length_at_switch':len(w.path),'start_cursor':list(w.cursor),'start_distance':distance(w.cursor,target),
      'menu':menu,'votes':votes,'best':best,'suffix':results}
    if not retain:
        for rr in rec['suffix'].values():
            for v in rr.values():v.pop('rows')
    return rec

def cluster_interval(records, metric, seed=44017):
    """Deterministic, paired world/seed/variant cluster bootstrap, descriptive."""
    groups={}
    for r in records:
        g=(r['seed'],r['spec']['family'],r['spec']['challenge'],r['variant'])
        groups.setdefault(g,[]).append(r)
    chunks=list(groups.values());rng=np.random.default_rng(seed)
    diffs=[]
    for _ in range(2000):
        sample=[r for j in rng.integers(0,len(chunks),len(chunks)) for r in chunks[j]]
        diffs.append(statistics.mean(metric(r) for r in sample))
    return [round(float(v),4) for v in np.quantile(diffs,[.025,.975])]

def aggregate(states):
    eligible=[r for r in states if not r['goal_already_ink']]
    per_arm={}
    for arm in ARMS:
        self_regrets=[];robust_regrets=[];regret_no_reloc=[];oracle_hits=0;policy_hits=0;different=0;first_actions=Counter();flips=0;oracle_first=Counter(); no_reloc_best=Counter();relocate_over_64=0
        for r in eligible:
            selected=r['votes'][arm]['chosen'];first_actions[selected]+=1
            oracle=r['suffix'][arm]
            oracle_first.update(r['best'][arm])
            allowed={a:v for a,v in oracle.items() if a!='RELOCATE_START'}
            no_reloc_best.update(best_action(allowed))
            if 'RELOCATE_START' in r['best'][arm] and r['path_length_at_switch']-1>64:
                relocate_over_64+=1
            if selected in allowed:
                regret_no_reloc.append(allowed[selected]['final_distance']-min(v['final_distance'] for v in allowed.values()))
            best_final=min(v['final_distance'] for v in oracle.values())
            self_regrets.append(oracle[selected]['final_distance']-best_final)
            if selected not in r['best'][arm]:different+=1
            if any(v['hit_step'] is not None for v in oracle.values()):oracle_hits+=1
            if oracle[selected]['hit_step'] is not None:policy_hits+=1
            for suffix_arm in ARMS:
                rr=r['suffix'][suffix_arm]
                robust_regrets.append(rr[selected]['final_distance']-min(v['final_distance'] for v in rr.values()))
            picks=set(tuple(r['best'][x]) for x in ARMS)
            if len(picks)>1:flips+=1
        per_arm[arm]={'n':len(eligible),'chosen_action_counts':dict(sorted(first_actions.items())),
          'mean_self_suffix_distance_regret':round(statistics.mean(self_regrets),4) if self_regrets else None,
          'mean_cross_suffix_distance_regret':round(statistics.mean(robust_regrets),4) if robust_regrets else None,
          'nonzero_self_distance_regrets':sum(v>0 for v in self_regrets),
          'mean_self_regret_excluding_relocate_oracle':round(statistics.mean(regret_no_reloc),4) if regret_no_reloc else None,
          'no_relocate_comparable_selected_cases':len(regret_no_reloc),
          'oracle_best_action_counts':dict(sorted(oracle_first.items())),
          'no_relocate_oracle_best_action_counts':dict(sorted(no_reloc_best.items())),
          'oracle_relocate_with_path_over_64':relocate_over_64,
          'nonoptimal_lexicographic_choices':different,'fork_has_any_goal_hit':oracle_hits,
          'chosen_first_goal_hit':policy_hits,'suffix_oracle_disagreement_states':flips}
    return {'total_switch_states':len(states),'eligible_goal_unpainted':len(eligible),
            'excluded_already_ink':len(states)-len(eligible),'arms':per_arm,
            'switch_vs_frozen_cross_suffix_regret_delta':round(statistics.mean(
                statistics.mean(r['suffix'][s][r['votes']['switch']['chosen']]['final_distance']
                       - r['suffix'][s][r['votes']['frozen']['chosen']]['final_distance'] for s in ARMS)
                        for r in eligible),4) if eligible else None,
            'switch_vs_frozen_delta_cluster_bootstrap_95pct':cluster_interval(eligible,lambda r: statistics.mean(
                r['suffix'][s][r['votes']['switch']['chosen']]['final_distance']
                -r['suffix'][s][r['votes']['frozen']['chosen']]['final_distance'] for s in ARMS)) if eligible else None,
            'by_switch':{str(k):{'total':sum(x['switch_step']==k for x in states),
                               'eligible':sum(x['switch_step']==k and not x['goal_already_ink'] for x in states)}
                          for k in prior.SWITCHES}}

def benchmark(models):
    """Real observed synthetic host-menu feature matrices; no synthetic timing claim."""
    matrices=[]
    sp=prior.dev_specs()[0]
    for seed in models:
        w=World.create(sp); goals=prior.goals_for(w,seed)
        for t in range(112):
            if w.stopped:break
            goal=goals[sum(t>=x for x in prior.SWITCHES)]
            menu,x=base.matrix(w,goal,160,False)
            if t in (3,24,55,85):matrices.append((seed,x,len(menu)))
            a=vote(models[seed]['frozen'],w,goal)['chosen'];w.apply(a,12)
    samples=[]
    for seed,x,n in matrices:
        for arm in ARMS:
            model=models[seed][arm]
            for i in range(128):
                begin=time.perf_counter_ns();model.probs(x,n)
                samples.append((time.perf_counter_ns()-begin)/1e6)
    return {'samples':len(samples), 'median_neural_softmax_ms':round(float(np.median(samples)),6),
            'p95_neural_softmax_ms':round(float(np.quantile(samples,.95)),6),
            'scope':'NumPy model probability evaluation only; excludes game update, rendering and IO'}

def render_gallery(states,path):
    """Six source-derived orthographic 2D diagnostic panels, not generated art."""
    eligible=[r for r in states if not r['goal_already_ink']]
    choices=[]
    for seed in SEEDS:
        subset=[r for r in eligible if r['seed']==seed]
        def score(r):
            choices_no_reloc={a:v for a,v in r['suffix']['frozen'].items() if a!='RELOCATE_START'}
            own=r['votes']['frozen']['chosen']
            gap=(choices_no_reloc[own]['final_distance']-min(v['final_distance'] for v in choices_no_reloc.values())) if own in choices_no_reloc else 0
            return (gap, int('RELOCATE_START' in r['best']['frozen']),r['switch_step'])
        selected=[];seen=set()
        for r in sorted(subset,key=score,reverse=True):
            if r['spec']['family'] in seen:continue
            selected.append(r);seen.add(r['spec']['family'])
            if len(selected)==3:break
        choices+=selected
    width=1000;height=765;svg=ET.Element('svg',{'xmlns':'http://www.w3.org/2000/svg',
      'viewBox':f'0 0 {width} {height}','width':str(width),'height':str(height)})
    ET.SubElement(svg,'rect',{'width':str(width),'height':str(height),'fill':'#0c1118'})
    for idx,r in enumerate(choices):
        x0=12+(idx%3)*331; y0=14+(idx//3)*362
        ET.SubElement(svg,'rect',{'x':str(x0),'y':str(y0),'width':'321','height':'352',
                                     'rx':'8','fill':'#15202c','stroke':'#344254'})
        sp=Spec(**r['spec']);w=World.create(sp)
        def tx(pt):return x0+15+int(pt[0])*3.02
        def ty(pt):return y0+36+int(pt[1])*2.95
        def xy_path(points):return ' '.join(f'{tx(a):.1f},{ty(a):.1f}' for a in points)
        for (xx,yy),mat in w.materials.items():
            ET.SubElement(svg,'rect',{'x':f'{tx((xx,yy)):.1f}','y':f'{ty((xx,yy)):.1f}',
                                        'width':'3','height':'3','fill':'#495464'})
        for a in w.protected:
            ET.SubElement(svg,'rect',{'x':f'{tx(a):.1f}','y':f'{ty(a):.1f}',
                                      'width':'3','height':'3','fill':'#923c66'})
        no_reloc={a:v for a,v in r['suffix']['frozen'].items() if a!='RELOCATE_START'}
        actions=[('frozen',r['votes']['frozen']['chosen'],'#5eead4'),
                 ('counterfactual',best_action(no_reloc)[0],'#f9bc68')]
        if 'RELOCATE_START' in r['menu']:
            actions.append(('relocate','RELOCATE_START','#e879f9'))
        for name,action,color in actions:
            fork=r['suffix']['frozen'][action]
            points=[r['start_cursor']]+[row['receipt']['cursor_after'] for row in fork['rows']]
            if action=='RELOCATE_START' and len(points)>1:
                ET.SubElement(svg,'polyline',{'points':xy_path(points[:2]),'fill':'none',
                          'stroke':color,'stroke-width':'1.35','stroke-dasharray':'4 3','opacity':'.9'})
                points=points[1:]
            ET.SubElement(svg,'polyline',{'points':xy_path(points),'fill':'none',
                        'stroke':color,'stroke-width':'2.1','opacity':'.92'})
        target=r['target']; start=r['start_cursor']
        ET.SubElement(svg,'circle',{'cx':str(tx(target)),'cy':str(ty(target)),'r':'5',
                                     'fill':'#ffffff','stroke':'#bf95ff','stroke-width':'2'})
        ET.SubElement(svg,'circle',{'cx':str(tx(start)),'cy':str(ty(start)),'r':'4.5','fill':'#fff4bf'})
        def text_line(y,content,font='12',fill='#dbeafe'):
            ET.SubElement(svg,'text',{'x':str(x0+12),'y':str(y0+y),'fill':fill,
                                     'font-family':'sans-serif','font-size':font}).text=content
        text_line(21,f"seed {r['seed']} · {sp.family}/{sp.challenge} · t={r['switch_step']}", '13')
        text_line(332,f"frozen {r['votes']['frozen']['chosen']}  |  no-reloc {actions[1][1]}", '11')
        text_line(346,f"target {target} · 16 decisions · synthetic",'10','#94a3b8')
    def legend(x,color,label):
        ET.SubElement(svg,'rect',{'x':str(x),'y':'744','width':'15','height':'3','fill':color})
        ET.SubElement(svg,'text',{'x':str(x+19),'y':'751','fill':'#dbeafe','font-size':'11','font-family':'sans-serif'}).text=label
    legend(15,'#5eead4','frozen choice');legend(180,'#f9bc68','best without relocation');legend(440,'#e879f9','relocation (dashed jump)')
    ET.ElementTree(svg).write(path,encoding='utf-8',xml_declaration=True)

def manifest(root):return {str(x.relative_to(root)):sha_file(x) for x in sorted(root.rglob('*')) if x.is_file() and x.name!='SHA256SUMS.json'}

def run(out,checkpoint_dir,seeds=SEEDS):
    if out.exists():raise FileExistsError('immutable output')
    out.mkdir(parents=True);(out/'states').mkdir();started=time.perf_counter()
    models,hashes=load_models(checkpoint_dir,seeds)
    records=[];prefixes=[]
    for seed in seeds:
        for sp in prior.dev_specs():
            for variant in (0,1):
                ws,prefix=prefix_states(sp,seed,variant,models[seed]['frozen'])
                prefixes.append({'seed':seed,'spec':sp.__dict__,'variant':variant,**prefix})
                for state in ws:
                    r=evaluate_state(state,models,seed,sp,variant)
                    (out/'states'/f'{len(records):04d}.json').write_text(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n')
                    records.append(r)
    rep={'version':VERSION,'source_sha256':sources(),'checkpoints':hashes,
         'split':'development','prefix_source':'frozen policy, on-policy, no forced moves',
         'suffix_lookahead':LOOKAHEAD,'switch_steps':list(prior.SWITCHES),
         'seeds':list(seeds),'structural_heldout_opened':False,'native_receipts':0,'paid_compute':False,
         'prefixes':prefixes,'summary':aggregate(records),
         'inference_benchmark':benchmark(models),
         'forks':sum(sum(len(z) for z in r['suffix'].values()) for r in records),
         'fork_decisions':sum(sum(len(v['rows']) for z in r['suffix'].values() for v in z.values()) for r in records),
         'wall_seconds':round(time.perf_counter()-started,3),
         'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    render_gallery(records,out/'TRAJECTORIES.svg')
    (out/'report.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps(manifest(out),indent=2,sort_keys=True)+'\n')
    return rep

def verify(out,checkpoint_dir):
    rep=json.loads((out/'report.json').read_text())
    if rep['version']!=VERSION or rep['source_sha256']!=sources() or rep['structural_heldout_opened']:
        raise ValueError('source drift or wrong split')
    if manifest(out)!=json.loads((out/'SHA256SUMS.json').read_text()):raise ValueError('manifest mismatch')
    models,hashes=load_models(checkpoint_dir,rep['seeds'])
    if hashes!=rep['checkpoints']:raise ValueError('checkpoint drift')
    expected=[];prefixes=[];n=0
    for seed in rep['seeds']:
        for sp in prior.dev_specs():
            for variant in (0,1):
                ws,prefix=prefix_states(sp,seed,variant,models[seed]['frozen'])
                prefixes.append({'seed':seed,'spec':sp.__dict__,'variant':variant,**prefix})
                for state in ws:
                    orig=json.loads((out/'states'/f'{n:04d}.json').read_text())
                    replayed=evaluate_state(state,models,seed,sp,variant)
                    if orig!=replayed:raise ValueError('neural/host counterfactual replay differs at '+str(n))
                    expected.append(orig);n+=1
    if n!=len(list((out/'states').glob('*.json'))) or rep['summary']!=aggregate(expected) or rep['prefixes']!=prefixes:
        raise ValueError('summary/prefix drift')
    forks=sum(sum(len(z) for z in r['suffix'].values()) for r in expected)
    decisions=sum(sum(len(v['rows']) for z in r['suffix'].values() for v in z.values()) for r in expected)
    if forks!=rep['forks'] or decisions!=rep['fork_decisions']:raise ValueError('fork count drift')
    return {'verified_states':n,'forks':forks,'decisions':decisions,'checkpoints':sum(map(len,models.values())),'files':len(manifest(out))}

def compact(out):
    """Entire state-by-state decision evidence, excluding long per-fork transcript arrays."""
    rep=json.loads((out/'report.json').read_text())
    es=[]
    for p in sorted((out/'states').glob('*.json')):
        r=json.loads(p.read_text())
        compact_states={}
        for suffix,rr in r['suffix'].items():
            compact_states[suffix]={a:{k:v for k,v in item.items() if k in
               ('final_distance','hit_step','added_recoveries','stopped','end_state_sha','trace_sha')}
               for a,item in rr.items()}
        es.append({'seed':r['seed'],'family':r['spec']['family'],'challenge':r['spec']['challenge'],
         'spec':r['spec']['seed'],'variant':r['variant'],'switch_step':r['switch_step'],
         'target':r['target'],'start_cursor':r['start_cursor'],'start_distance':r['start_distance'],
         'prefix_hash':r['prefix_hash'],'already_ink':r['goal_already_ink'],'menu':r['menu'],
         'votes':r['votes'],'oracles':r['best'],'fork_outcomes':compact_states})
    return {'version':VERSION,'source_sha256':rep['source_sha256'],'checkpoints':rep['checkpoints'],
            'partition':rep['split'],'prefixes':rep['prefixes'],'summary':rep['summary'],
            'inference_benchmark':rep['inference_benchmark'],
            'forks':rep['forks'],'fork_decisions':rep['fork_decisions'],
            'local_manifest_sha256':sha_file(out/'SHA256SUMS.json'),'states':es}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--checkpoints',required=True,type=Path)
    p.add_argument('--verify',type=Path);p.add_argument('--compact',type=Path);p.add_argument('--quick',action='store_true')
    a=p.parse_args()
    if a.verify: print(json.dumps(verify(a.verify,a.checkpoints)))
    elif a.compact: a.compact.write_text(json.dumps(compact(a.out),sort_keys=True,indent=2)+'\n')
    else:
        x=run(a.out,a.checkpoints,(619,) if a.quick else SEEDS)
        print(json.dumps({k:v for k,v in x.items() if k not in ('prefixes','checkpoints','source_sha256')},sort_keys=True))
if __name__=='__main__':main()
