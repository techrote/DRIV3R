"""RETRACE-REALISM-001: matched, relocation-free first-action counterfactuals.

Synthetic CyberSand-inspired fixture only. No native ABI claim. This diagnostic
never changes World.menu(), never asks World.apply() to execute a nonlegal
command, and does not modify the historical d_common fixture. It restricts the
*research policy's choices* to movement/BACKTRACK/STOP while recording the full
synthetic host menu and neural probabilities. An explicit 64-logical-move proxy
censors synthetic continuations; it is not native CyberSand execution.

No new checkpoints are trained and structural heldout families are never used.
"""
from __future__ import annotations

import argparse
import copy
import json
import resource
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from . import goal_recovery_lab as base
from . import goal_switch_curriculum_lab as prior
from . import post_switch_regret_lab as previous
from .world import Spec, World, MOVE, digest
from .train import sha_file

VERSION = 'retrace-realism/1'
SEEDS = (619, 620)
ARMS = ('frozen', 'sham', 'switch')
HORIZON = 32
MOVE_BUDGET_PROXY = 64
FIRST_SWITCH = 32
SOURCE_NAMES = ('retrace_realism_lab.py', 'post_switch_regret_lab.py',
                'goal_switch_curriculum_lab.py', 'goal_recovery_lab.py',
                'goal_retarget_lab.py', 'goal_conditioning_lab.py',
                'policy.py', 'world.py', 'train.py')


def source_hashes():
    return {name: sha_file(Path(__file__).parent / name) for name in SOURCE_NAMES}


def specimen_specs(cohort):
    if cohort == 'previous_development':
        return prior.dev_specs()
    if cohort == 'fresh_development':
        # Predeclared different DEVELOPMENT seeds, same four known TRAIN families.
        # These are not the structurally disjoint sealed HELDOUT families.
        pairs = (('pickets', 3007), ('ridges', 3107),
                 ('islands', 3207), ('zigzag', 3307))
        return [replace(Spec('development', family, start + offset,
                             ('wall', 'route', 'motif')[(start + offset) % 3]),
                        challenge=challenge)
                for family, start in pairs for offset in (0, 1)
                for challenge in ('normal', 'trap_corridor')]
    raise ValueError('unknown cohort')


def start_state(sp, seed, variant, model):
    """Only the first target-switch state, at decision 32, on real model actions."""
    states, prefix = previous.prefix_states(sp, seed, variant, model)
    state = next((s for s in states if s['step'] == FIRST_SWITCH), None)
    if state is None:
        return None, prefix
    w = state['world']
    if (w.steps != FIRST_SWITCH or w.logical_moves != FIRST_SWITCH or
        any(d['chosen'] == 'RELOCATE_START' for d in w.decisions)):
        return None, {**prefix,'excluded_earlier_relocation':True}
    return state, {**prefix,'excluded_earlier_relocation':False}


def conditional_choice(menu, probabilities):
    permitted = [a for a in menu if a != 'RELOCATE_START']
    if not permitted:
        raise ValueError('no non-relocation action')
    # Stable menu-order tie break, same argmax principle as the original model.
    return max(permitted, key=lambda a: probabilities[menu.index(a)])


def fork(w, first, goal, model):
    w = copy.deepcopy(w)
    initial_recoveries = w.recoveries
    t0 = w.logical_moves
    rows = []
    hit_step = None
    for i in range(HORIZON):
        if w.stopped or w.logical_moves >= MOVE_BUDGET_PROXY:
            break
        menu, x = base.matrix(w, goal, 160, False)
        probs = model.probs(x, len(menu))
        if menu != w.menu(12) or len(probs) != len(menu) or abs(sum(probs)-1) > 1e-8:
            raise ValueError('native-inspired synthetic host menu or probabilities changed')
        natural = menu[int(np.argmax(probs))]
        permitted = conditional_choice(menu, probs)
        selected = first if i == 0 else permitted
        if selected not in menu or selected == 'RELOCATE_START':
            raise ValueError('research policy issued disallowed action')
        before = w.cursor
        receipt = w.apply(selected, 12)
        if selected == 'BACKTRACK' and abs(before[0]-w.cursor[0])+abs(before[1]-w.cursor[1]) != 1:
            raise ValueError('backtracking was not cardinal/adjacent')
        if w.logical_moves > MOVE_BUDGET_PROXY:
            raise ValueError('synthetic logical-move proxy exceeded')
        if w.cursor == goal and hit_step is None:
            hit_step = i+1
        rows.append({'host_menu': menu, 'probabilities': [round(float(p), 10) for p in probs],
                     'raw_argmax': natural, 'permitted_argmax': permitted,
                     'executed': selected, 'receipt': receipt})
    return {
        'first_action': first,
        'goal_hit_step': hit_step,
        'final_distance': previous.distance(w.cursor, goal),
        'final_cursor': list(w.cursor),
        'final_state_sha': w.state_sha(),
        'trace_sha': digest(w.decisions),
        'added_recoveries': w.recoveries-initial_recoveries,
        'new_ink': sum(len(x['receipt']['painted']) for x in rows),
        'logical_moves_added': w.logical_moves-t0,
        'budget_censored': w.logical_moves == MOVE_BUDGET_PROXY and not w.stopped,
        'stopped': w.stopped,
        'masked_relocation_count': sum(x['raw_argmax'] == 'RELOCATE_START' and
                                       x['executed'] != 'RELOCATE_START' for x in rows[1:]),
        'rows': rows,
    }


def outcome_key(fork_result):
    hit = fork_result['goal_hit_step']
    return (int(hit is not None), -hit if hit is not None else -fork_result['final_distance'])


def robust_margin(backtrack_result, best_forward):
    """Strict improvement for all arms, and material margin: >=4 steps if both hit,
    or >=8 final-distance cells if neither hits, or hit versus no-hit.
    """
    b, f = backtrack_result, best_forward
    bh, fh = b['goal_hit_step'], f['goal_hit_step']
    if bh is not None and fh is None:
        return True
    if bh is not None and fh is not None:
        return fh - bh >= 4
    if bh is None and fh is None:
        return f['final_distance']-b['final_distance'] >= 8
    return False


def evaluate_state(state, models, seed, sp, variant, cohort):
    w = state['world']
    goal = state['goal']
    menu = w.menu(12)
    candidates = [a for a in menu if a in MOVE or a == 'BACKTRACK']
    if not candidates or 'BACKTRACK' not in candidates:
        raise ValueError('no paired retreat and forward candidates')
    if not any(x in MOVE for x in candidates):
        raise ValueError('no forward comparison')
    votes = {arm: previous.vote(models[seed][arm], w, goal) for arm in ARMS}
    outcomes = {arm: {a: fork(w, a, goal, models[seed][arm]) for a in candidates}
                for arm in ARMS}
    per_arm = {}
    for arm in ARMS:
        bt = outcomes[arm]['BACKTRACK']
        best_move_name = max((a for a in candidates if a in MOVE),
                             key=lambda a: outcome_key(outcomes[arm][a]))
        forward = outcomes[arm][best_move_name]
        per_arm[arm] = {
            'best_forward': best_move_name,
            'retreat_strict_win': outcome_key(bt) > outcome_key(forward),
            'retreat_material_win': robust_margin(bt, forward),
            'retreat_hit': bt['goal_hit_step'],
            'best_forward_hit': forward['goal_hit_step'],
            'retreat_final_distance': bt['final_distance'],
            'best_forward_final_distance': forward['final_distance']}
    return {'cohort': cohort, 'seed': seed, 'spec': sp.__dict__,
            'variant': variant, 'switch_at': state['step'],
            'prefix_hash': state['prefix_hash'], 'base_sha': state['base_hash'],
            'start_state_sha': w.state_sha(), 'start_cursor': list(w.cursor),
            'goal': list(goal), 'goal_prepainted': goal in w.ink,
            'prehistory_logical_moves': w.logical_moves, 'prehistory_recoveries': w.recoveries,
            'host_menu': menu, 'candidates': candidates, 'raw_votes': votes,
            'outcomes': outcomes, 'comparisons': per_arm,
            'all_strict': all(p['retreat_strict_win'] for p in per_arm.values()),
            'all_material': all(p['retreat_material_win'] for p in per_arm.values())}


def aggregate(rows):
    eligible = [r for r in rows if not r['goal_prepainted']]
    out = {'states': len(rows), 'eligible': len(eligible), 'prepainted': len(rows)-len(eligible),
           'conditions': {}, 'arms': {}, 'candidate_distribution': {}, 'strict_robust': [],
           'material_robust': [], 'source': 'first-switch prehistory; synthetic non-relocation action constraint'}
    for cohort in ('previous_development','fresh_development'):
        subset = [r for r in eligible if r['cohort'] == cohort]
        out['conditions'][cohort] = {
            'eligible': len(subset), 'all_strict': sum(r['all_strict'] for r in subset),
            'all_material': sum(r['all_material'] for r in subset),
            'per_seed_material': {str(seed):sum(r['all_material'] for r in subset if r['seed']==seed)
                                  for seed in SEEDS}}
    count=Counter()
    for r in eligible: count.update(r['candidates'])
    out['candidate_distribution']=dict(sorted(count.items()))
    for arm in ARMS:
        a = [r for r in eligible]
        forks = [f for r in rows for f in r['outcomes'][arm].values()]
        out['arms'][arm] = {
            'first_choice_counts': dict(sorted(Counter(r['raw_votes'][arm]['chosen'] for r in a).items())),
            'retreat_strict_preference': sum(r['comparisons'][arm]['retreat_strict_win'] for r in a),
            'retreat_material_preference': sum(r['comparisons'][arm]['retreat_material_win'] for r in a),
            'forks': len(forks), 'decisions': sum(len(f['rows']) for f in forks),
            'budget_censored': sum(f['budget_censored'] for f in forks),
            'relocation_prevented': sum(f['masked_relocation_count'] for f in forks),
            'goal_hits': sum(f['goal_hit_step'] is not None for f in forks),
            'stops': sum(f['stopped'] for f in forks),
            'model_first_relocation': sum(r['raw_votes'][arm]['chosen']=='RELOCATE_START' for r in a),
            'executed_action_distribution':dict(sorted(Counter(z['executed'] for f in forks
                                                        for z in f['rows']).items()))}
    for r in eligible:
        key = {'cohort':r['cohort'],'seed':r['seed'],'family':r['spec']['family'],
               'world_seed':r['spec']['seed'], 'challenge':r['spec']['challenge'],
               'variant':r['variant'],'prefix_hash':r['prefix_hash'],
               'raw_votes':{k:v['chosen'] for k,v in r['raw_votes'].items()},
               'comparisons':r['comparisons']}
        if r['all_strict']:out['strict_robust'].append(key)
        if r['all_material']:out['material_robust'].append(key)
    return out


def manifest(out):
    return {str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*'))
            if p.is_file() and p.name!='SHA256SUMS.json'}


def gallery(records, path):
    candidates = [r for r in records if not r['goal_prepainted']]
    # A fixed, stated ranking *after* results: illustrative only, never a test statistic.
    chosen = sorted(candidates, key=lambda r:(not r['all_material'], not r['all_strict'],
                   r['cohort'] != 'fresh_development',r['seed'],r['spec']['seed'],r['variant']))[:6]
    width,height=1080,800
    svg=ET.Element('svg',{'xmlns':'http://www.w3.org/2000/svg', 'viewBox':f'0 0 {width} {height}'})
    ET.SubElement(svg,'rect',{'width':str(width),'height':str(height),'fill':'#0e1420'})
    title=ET.SubElement(svg,'text',{'x':'14','y':'18','fill':'#eaf2ff','font-size':'14',
                                   'font-family':'sans-serif'})
    title.text='RETRACE-REALISM-001 • original synthetic source replay • 32-step fork / 64-logical-move proxy'
    for index,r in enumerate(chosen):
        left=12+(index%3)*356; top=34+(index//3)*376
        ET.SubElement(svg,'rect',{'x':str(left),'y':str(top),'width':'345','height':'360',
                                  'rx':'7','fill':'#18263a','stroke':'#36455d'})
        spec=Spec(**r['spec']); world=World.create(spec)
        def coord(pt):return (left+11+pt[0]*3.27,top+34+pt[1]*3.14)
        def line(points):return ' '.join('%.1f,%.1f'%coord(p) for p in points)
        for p in world.materials:
            x,y=coord(p); ET.SubElement(svg,'rect',{'x':str(x),'y':str(y), 'width':'3.2','height':'3.1','fill':'#65758c'})
        for p in world.protected:
            x,y=coord(p); ET.SubElement(svg,'rect',{'x':str(x),'y':str(y),'width':'3.2','height':'3.1','fill':'#b85a79'})
        arm='frozen'
        bt=r['outcomes'][arm]['BACKTRACK']
        best=r['comparisons'][arm]['best_forward']
        fwd=r['outcomes'][arm][best]
        for name,rr,c in [('BACKTRACK',bt,'#71e5bf'),(best,fwd,'#f5b45c')]:
            pts=[r['start_cursor']]+[x['receipt']['cursor_after'] for x in rr['rows']]
            ET.SubElement(svg,'polyline',{'points':line(pts),'stroke':c,'fill':'none',
                                          'stroke-width':'2.4','stroke-linejoin':'round','opacity':'.90'})
        gx,gy=coord(r['goal']);sx,sy=coord(r['start_cursor'])
        ET.SubElement(svg,'circle',{'cx':str(gx),'cy':str(gy),'r':'5','fill':'#e5dbff','stroke':'#815ccd'})
        ET.SubElement(svg,'circle',{'cx':str(sx),'cy':str(sy),'r':'4','fill':'#fff4bf'})
        def label(y, txt, size=11, color='#d5e3f6'):
            ET.SubElement(svg,'text',{'x':str(left+11),'y':str(top+y),'fill':color,
                                     'font-size':str(size),'font-family':'sans-serif'}).text=txt
        label(18,f"{r['cohort'].replace('_development','')} • {r['seed']} • {spec.family}/{spec.challenge}",12)
        label(335,f"retreat: hit {bt['goal_hit_step']}  distance {bt['final_distance']}")
        label(350,f"{best}: hit {fwd['goal_hit_step']}  distance {fwd['final_distance']}")
    ET.SubElement(svg,'text',{'x':'15','y':'790','font-family':'sans-serif','font-size':'11',
                              'fill':'#aebfd4'}).text=(
      'Teal: adjacent BACKTRACK first; amber: best MOVE first. Targets white; start cream. '
      'No native execution.')
    ET.ElementTree(svg).write(path,encoding='utf-8',xml_declaration=True)


def benchmark(models):
    # Model-only inference timing; excludes environment updates, map rendering and I/O.
    w = World.create(Spec('development','pickets',3002,'wall'))
    menu, features = base.matrix(w,(53,44),160,False)
    samples = {}
    for arm in ARMS:
        model = models[619][arm]
        for _ in range(32): model.probs(features,len(menu))
        lat = []
        for _ in range(512):
            t0 = time.perf_counter_ns()
            model.probs(features,len(menu))
            lat.append((time.perf_counter_ns()-t0)*1e-6)
        samples[arm] = {'median_ms':round(float(np.median(lat)),6),
                        'p95_ms':round(float(np.quantile(lat,.95)),6),
                        'calls':len(lat),'scope':'model probability vector only'}
    return samples


def run(out, checkpoint_dir, cohorts=('previous_development','fresh_development')):
    if out.exists(): raise FileExistsError('immutable experiment output exists')
    out.mkdir(parents=True); (out/'states').mkdir()
    started=time.perf_counter()
    models,hashes=previous.load_models(checkpoint_dir,SEEDS)
    states=[]; prefixes=[]
    for cohort in cohorts:
        for seed in SEEDS:
            for sp in specimen_specs(cohort):
                for variant in (0,1):
                    s,prefix = start_state(sp,seed,variant,models[seed]['frozen'])
                    prefixes.append({'cohort':cohort,'seed':seed,'spec':sp.__dict__,
                                     'variant':variant,**prefix})
                    if s is None: continue
                    r=evaluate_state(s,models,seed,sp,variant,cohort)
                    (out/'states'/f'{len(states):04d}.json').write_text(
                      json.dumps(r,sort_keys=True,separators=(',',':'))+'\n')
                    states.append(r)
    result = {'version':VERSION, 'cohorts':list(cohorts), 'seeds':list(SEEDS),
              'sources_sha256':source_hashes(),'checkpoint_sha256':hashes,
              'horizon':HORIZON,'synthetic_logical_move_proxy':MOVE_BUDGET_PROXY,
              'first_switch':FIRST_SWITCH,'prefixes':prefixes,
              'excluded_prehistories':sum(p.get('excluded_earlier_relocation',False)
                                         for p in prefixes),
              'model_only_latency':benchmark(models),
              'summary':aggregate(states),
              'structural_heldout_opened':False,'native_receipts':0,
              'paid_compute':False,'new_optimizer_updates':0,
              'elapsed_s':round(time.perf_counter()-started,3),
              'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    gallery(states,out/'TRAJECTORIES.svg')
    (out/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps(manifest(out),sort_keys=True,indent=2)+'\n')
    return result


def verify(out, checkpoints):
    report=json.loads((out/'report.json').read_text())
    if (report['version'] != VERSION or report['sources_sha256'] != source_hashes() or
        report['structural_heldout_opened'] or report['new_optimizer_updates'] != 0):
        raise ValueError('wrong version, source drift or improper split')
    if manifest(out) != json.loads((out/'SHA256SUMS.json').read_text()):
        raise ValueError('evidence manifest mismatch')
    models,sha = previous.load_models(checkpoints,SEEDS)
    if report['checkpoint_sha256'] != sha:raise ValueError('checkpoint drift')
    states=[]; prefixes=[]
    for cohort in report['cohorts']:
        for seed in SEEDS:
            for sp in specimen_specs(cohort):
                for variant in (0,1):
                    s,prefix = start_state(sp,seed,variant,models[seed]['frozen'])
                    prefixes.append({'cohort':cohort,'seed':seed,'spec':sp.__dict__,
                                     'variant':variant,**prefix})
                    if s is None:continue
                    reconstructed=evaluate_state(s,models,seed,sp,variant,cohort)
                    evidence=json.loads((out/'states'/f'{len(states):04d}.json').read_text())
                    if reconstructed != evidence:
                        raise ValueError('source-matched counterfactual does not replay at '+str(len(states)))
                    states.append(reconstructed)
    if (report['prefixes'] != prefixes or report['summary'] != aggregate(states) or
        report['excluded_prehistories'] != sum(p.get('excluded_earlier_relocation',False) for p in prefixes) or
        len(states) != len(list((out/'states').glob('*.json')))):
        raise ValueError('summary/prefix/count mismatch')
    count = sum(len(f['rows']) for r in states for arms in r['outcomes'].values() for f in arms.values())
    return {'verified_states':len(states),'forks':sum(len(a) for r in states for a in r['outcomes'].values()),
            'verified_host_legal_decisions':count,'checkpoints':sum(len(v) for v in models.values()),
            'manifested_files':len(manifest(out))}


def compact(out):
    rep=json.loads((out/'report.json').read_text())
    rows=[]
    for p in sorted((out/'states').glob('*.json')):
        r=json.loads(p.read_text())
        cases={arm:{act:{k:value for k,value in f.items() if k!='rows'}
                    for act,f in arms.items()} for arm,arms in r['outcomes'].items()}
        rows.append({k:v for k,v in r.items() if k not in ('outcomes','raw_votes')} |
                    {'first_action_probabilities':r['raw_votes'],'outcomes':cases})
    return {'version':VERSION,'sources_sha256':rep['sources_sha256'],
            'checkpoint_sha256':rep['checkpoint_sha256'],
            'synthetic_logical_move_proxy':MOVE_BUDGET_PROXY,
            'summary':rep['summary'], 'case_matrix':rows,
            'local_manifest_sha256':sha_file(out/'SHA256SUMS.json')}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--checkpoints',required=True,type=Path)
    ap.add_argument('--verify',action='store_true');ap.add_argument('--compact',type=Path)
    a=ap.parse_args()
    if a.verify:
        print(json.dumps(verify(a.out,a.checkpoints),sort_keys=True))
    elif a.compact:
        a.compact.write_text(json.dumps(compact(a.out),sort_keys=True,indent=2)+'\n')
    else:
        r=run(a.out,a.checkpoints)
        print(json.dumps({'summary':r['summary'],'elapsed_s':r['elapsed_s'],
                          'peak_rss_kib':r['peak_rss_kib']},sort_keys=True))

if __name__=='__main__':main()
