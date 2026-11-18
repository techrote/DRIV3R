"""RETRACE-SEED-TRANSFER-001: cross-seed suffix audit of rootless recovery.

Post-hoc robustness diagnostic on DEVELOPMENT cases from RETRACE-REALISM-001.
Every case keeps its original, verified frozen prefix, goal, legal actions, and
64-logical-move synthetic proxy. Each fork is replayed with six independent
continuation configurations (3 arms x 2 seeds). No optimizer updates, no
structural HELDOUT, no native CyberSand / production host-contract claims.
"""
from __future__ import annotations
import argparse, json, resource, time
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from . import retrace_realism_lab as prior
from . import post_switch_regret_lab as previous
from . import goal_recovery_lab as goal
from .world import Spec, MOVE, World
from .train import sha_file

VERSION='retrace-seed-transfer/1'
SOURCES=('retrace_seed_transfer_lab.py','retrace_realism_lab.py',
         'post_switch_regret_lab.py','goal_switch_curriculum_lab.py',
         'goal_recovery_lab.py','goal_retarget_lab.py',
         'goal_conditioning_lab.py','policy.py','world.py','train.py')


def pins():
    return {n:sha_file(Path(__file__).parent/n) for n in SOURCES}


def manifest(folder):
    return {str(p.relative_to(folder)):sha_file(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.name!='SHA256SUMS.json'}


def verify_original(folder):
    expected=json.loads((folder/'SHA256SUMS.json').read_text())
    if manifest(folder)!=expected: raise ValueError('original evidence checksum mismatch')
    return sha_file(folder/'SHA256SUMS.json')


def compare(state, models, old):
    sp=Spec(**old['spec']); seed=old['seed']
    rec={'cohort':old['cohort'],'seed':seed,'spec':old['spec'],
         'variant':old['variant'],'prefix_hash':old['prefix_hash'],
         'start_state_sha':old['start_state_sha'],'goal':old['goal'],
         'goal_prepainted':old['goal_prepainted'],'host_menu':old['host_menu'],
         'original_all_material':old['all_material'],
         'original_all_strict':old['all_strict'], 'suffixes':{}}
    w=state['world']; target=state['goal']
    assert state['prefix_hash']==old['prefix_hash']
    assert w.state_sha()==old['start_state_sha']
    assert list(target)==old['goal'] and w.menu(12)==old['host_menu']
    assert w.logical_moves==prior.FIRST_SWITCH
    acts=[a for a in rec['host_menu'] if a in MOVE or a=='BACKTRACK']
    assert 'BACKTRACK' in acts and any(a in MOVE for a in acts)
    total=0
    for suffix_seed in prior.SEEDS:
        for arm in prior.ARMS:
            name=f'{suffix_seed}-{arm}'
            model=models[suffix_seed][arm]
            vote=previous.vote(model,w,target)
            outcomes={action:prior.fork(w,action,target,model) for action in acts}
            best=max((a for a in acts if a in MOVE),key=lambda a:prior.outcome_key(outcomes[a]))
            retreat=outcomes['BACKTRACK']; forward=outcomes[best]
            strict=prior.outcome_key(retreat)>prior.outcome_key(forward)
            material=prior.robust_margin(retreat,forward)
            if material and not strict: raise ValueError('material without strict improvement')
            total+=sum(len(x['rows']) for x in outcomes.values())
            rec['suffixes'][name]={'source_seed':suffix_seed,'arm':arm,
              'first_action_probs':vote,'best_forward':best,
              'strict':strict,'material':material,
              'outcomes':outcomes}
    rec['all_six_strict']=all(x['strict'] for x in rec['suffixes'].values())
    rec['all_six_material']=all(x['material'] for x in rec['suffixes'].values())
    rec['suffix_opposition']=any(x['strict'] for x in rec['suffixes'].values()) and any(
                              not x['strict'] for x in rec['suffixes'].values())
    rec['decisions']=total
    return rec


def summary(rows):
    elig=[r for r in rows if not r['goal_prepainted']]
    cases=[r for r in elig if r['original_all_material']]
    a=Counter(); forking=Counter(); suites={}
    for r in rows:
        for key,v in r['suffixes'].items():
            for action,out in v['outcomes'].items():
                a.update(x['executed'] for x in out['rows'])
                forking['forks']+=1
                forking['decisions']+=len(out['rows'])
                forking['stops']+=int(out['stopped'])
                forking['goal_hits']+=int(out['goal_hit_step'] is not None)
                forking['budget_censored']+=int(out['budget_censored'])
                forking['relocations_prevented']+=out['masked_relocation_count']
    for seed in prior.SEEDS:
        for arm in prior.ARMS:
            key=f'{seed}-{arm}'
            suites[key]={'strict':sum(r['suffixes'][key]['strict'] for r in elig),
                         'material':sum(r['suffixes'][key]['material'] for r in elig),
                         'first_vote':dict(sorted(Counter(r['suffixes'][key]['first_action_probs']['chosen'] for r in elig).items()))}
    def aggregate(subset):
        return {'eligible':len(subset),'original_material':sum(r['original_all_material'] for r in subset),
                'six_strict':sum(r['all_six_strict'] for r in subset),
                'six_material':sum(r['all_six_material'] for r in subset),
                'suffix_opposition':sum(r['suffix_opposition'] for r in subset),
                'original_material_survived':sum(r['all_six_material'] for r in subset if r['original_all_material'])}
    return {'total_states':len(rows),'overall':aggregate(elig),
            'by_cohort':{c:aggregate([r for r in elig if r['cohort']==c])
                         for c in ('previous_development','fresh_development')},
            'by_prefix_seed':{str(s):aggregate([r for r in elig if r['seed']==s]) for s in prior.SEEDS},
            'suffix':suites,'forks':dict(sorted(forking.items())),
            'executed_action_distribution':dict(sorted(a.items())),
            'original_material_cases':[
                {'cohort':r['cohort'],'prefix_seed':r['seed'],'family':r['spec']['family'],
                 'world_seed':r['spec']['seed'],'variant':r['variant'],
                 'suffix_material':{k:v['material'] for k,v in r['suffixes'].items()},
                 'suffix_strict':{k:v['strict'] for k,v in r['suffixes'].items()}}
                for r in cases]}


def benchmark(models):
    w=World.create(Spec('development','pickets',3002,'wall'))
    menu,x=goal.matrix(w,(55,55),160,False)
    stats={}
    for seed in prior.SEEDS:
        for arm in prior.ARMS:
            model=models[seed][arm]
            for _ in range(32):model.probs(x,len(menu))
            lat=[]
            for _ in range(256):
                t0=time.perf_counter_ns();model.probs(x,len(menu));lat.append((time.perf_counter_ns()-t0)/1e6)
            stats[f'{seed}-{arm}']={'median_ms':round(float(np.median(lat)),6),
                                      'p95_ms':round(float(np.quantile(lat,.95)),6),
                                      'calls':len(lat),'scope':'model probabilities only'}
    return stats



def gallery(rows, output):
    """Six actual synthetic suffix trajectories; positions come from legal receipts."""
    selected=sorted((r for r in rows if r['all_six_material']),
                    key=lambda r:(r['cohort'],r['spec']['family'],r['spec']['seed'],r['variant']))
    controls=[r for r in rows if r['seed']==619 and not r['all_six_material'] and
              r['cohort']=='fresh_development']
    if len(selected)!=5 or not controls: raise ValueError('unexpected gallery cohort')
    selected.append(controls[0])
    svg=ET.Element('svg',{'xmlns':'http://www.w3.org/2000/svg','viewBox':'0 0 1110 800'})
    ET.SubElement(svg,'rect',{'width':'1110','height':'800','fill':'#101826'})
    ET.SubElement(svg,'text',{'x':'20','y':'23','fill':'#f2f6fb','font-size':'15',
        'font-family':'sans-serif'}).text='DRIV3R RETRACE-SEED-TRANSFER  •  same first-action fork, two independent suffix seeds'
    def label(x,y,st,color='#d9e4f0',size=11):
        ET.SubElement(svg,'text',{'x':str(x),'y':str(y),'fill':color,'font-size':str(size),
                                 'font-family':'sans-serif'}).text=st
    for i,r in enumerate(selected):
        left=15+(i%3)*365;top=40+(i//3)*376
        ET.SubElement(svg,'rect',{'x':str(left),'y':str(top),'width':'354','height':'365',
                                  'rx':'7','fill':'#1c2a40','stroke':'#50627a'})
        sp=Spec(**r['spec']);w=World.create(sp)
        def coord(pt):return (left+10+pt[0]*3.29,top+33+pt[1]*3.05)
        def points(pts):return ' '.join('%.1f,%.1f'%coord(pt) for pt in pts)
        for pos in w.materials:
            x,y=coord(pos);ET.SubElement(svg,'rect',{'x':str(x),'y':str(y),'width':'3','height':'3','fill':'#66758b'})
        for pos in w.protected:
            x,y=coord(pos);ET.SubElement(svg,'rect',{'x':str(x),'y':str(y),'width':'3','height':'3','fill':'#bd6889'})
        label(left+10,top+19,f"{r['cohort'].split('_')[0]} | prefix {r['seed']} | {sp.family}/{sp.challenge} | {sp.seed}",size=11)
        for suffix,color_good,color_bad in [('619-frozen','#55dfb1','#f9ba71'),
                                            ('620-frozen','#66a9f2','#e47591')]:
            entry=r['suffixes'][suffix];best=entry['best_forward']
            for action,clr in [('BACKTRACK',color_good),(best,color_bad)]:
                o=entry['outcomes'][action]
                receipts=[x['receipt']['cursor_after'] for x in o['rows']]
                start=o['rows'][0]['receipt']['cursor_before'] if o['rows'] else None
                if start is None:continue
                poly=ET.SubElement(svg,'polyline',{'points':points([start]+receipts),
                                    'stroke':clr,'fill':'none','stroke-width':'2.1',
                                    'stroke-linecap':'round','stroke-linejoin':'round'})
                if action!='BACKTRACK':poly.set('stroke-dasharray','4,3')
        gx,gy=coord(r['goal']);ET.SubElement(svg,'circle',{'cx':str(gx),'cy':str(gy),
            'r':'5.2','fill':'#f0d8fd','stroke':'#8666ae','stroke-width':'2'})
        label(left+10,top+336,('ROBUST across 6 suffixes' if r['all_six_material'] else 'CONTROL: no robust benefit'),
              '#8ceccc' if r['all_six_material'] else '#f4b886',12)
        label(left+10,top+353,f"Green/blue BACKTRACK; orange/red best forward | seed {r['seed']}",size=10)
    label(20,797,'Synthetic DEVELOPMENT worlds only. Dashed lines are MOVE-first alternatives; no native worm receipt.', '#afc0d8',10)
    ET.ElementTree(svg).write(output,encoding='utf-8',xml_declaration=True)


def new_evidence(checkpoint_dir, historical_dir, out, resume=False):
    if out.exists() and not resume: raise FileExistsError(out)
    if resume and (out/'report.json').exists(): raise FileExistsError('cannot resume sealed report')
    started=time.perf_counter(); historical_sha=verify_original(historical_dir)
    models,cksha=previous.load_models(checkpoint_dir,prior.SEEDS)
    if not resume:
        out.mkdir(parents=True);(out/'states').mkdir()
    rows=[]
    for src in sorted((historical_dir/'states').glob('*.json')):
        old=json.loads(src.read_text())
        sp=Spec(**old['spec'])
        state,prefix=prior.start_state(sp,old['seed'],old['variant'],models[old['seed']]['frozen'])
        if state is None:raise ValueError('original prefix missing')
        dst=out/'states'/src.name
        if resume and dst.is_file():
            rec=json.loads(dst.read_text())
            if rec['start_state_sha']!=old['start_state_sha'] or rec['prefix_hash']!=old['prefix_hash']:
                raise ValueError('incompatible resumed prefix evidence')
        else:
            rec=compare(state,models,old)
            dst.write_text(json.dumps(rec,sort_keys=True,separators=(',',':'))+'\n')
        rows.append(rec)
    report={'version':VERSION,'historical_manifest_sha256':historical_sha,
            'source_sha256':pins(),'checkpoint_sha256':cksha,
            'model_only_latency':benchmark(models),
            'horizon':prior.HORIZON,'synthetic_logical_move_proxy':prior.MOVE_BUDGET_PROXY,
            'summary':summary(rows),'elapsed_s':None if resume else round(time.perf_counter()-started,3),
            'last_resume_finalize_s':round(time.perf_counter()-started,3) if resume else None,
            'elapsed_note':'Interrupted/resumed development campaign; no uninterrupted total wall clock' if resume else 'single uninterrupted wall clock',
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'sealed_heldout_opened':False,'new_optimizer_updates':0,'native_receipts':0,
            'paid_compute':False}
    gallery(rows,out/'RETRACE_SEED_TRANSFER_TRAJECTORIES_2026-11-18.svg')
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps(manifest(out),sort_keys=True,indent=2)+'\n')
    return report


def verify(out, checkpoint_dir, historical_dir, sample=False):
    report=json.loads((out/'report.json').read_text())
    if (report['version']!=VERSION or report['source_sha256']!=pins() or
        report['historical_manifest_sha256']!=verify_original(historical_dir) or
        manifest(out)!=json.loads((out/'SHA256SUMS.json').read_text())):
        raise ValueError('source/manifest mismatch')
    models,cksha=previous.load_models(checkpoint_dir,prior.SEEDS)
    if cksha!=report['checkpoint_sha256']:raise ValueError('checkpoint drift')
    rows=[]
    allpaths=sorted((historical_dir/'states').glob('*.json'))
    for index,oldpath in enumerate(allpaths):
        old=json.loads(oldpath.read_text())
        if sample and not (old['all_material'] or index%12==0):continue
        sp=Spec(**old['spec'])
        state,_=prior.start_state(sp,old['seed'],old['variant'],models[old['seed']]['frozen'])
        if state is None:raise ValueError('prefix missing')
        got=compare(state,models,old)
        ref=json.loads((out/'states'/oldpath.name).read_text())
        if got!=ref:raise ValueError(f'suffix replay diverged: {oldpath.name}')
        rows.append(got)
    if not sample and summary(rows)!=report['summary']:raise ValueError('summary drift')
    return {'replayed_states':len(rows),'total_recorded_states':len(allpaths),
            'replay_scope':'material-positive-and-stratified-sample' if sample else 'all',
            'recorded_forks':report['summary']['forks']['forks'],
            'recorded_decisions':report['summary']['forks']['decisions'],
            'checkpoints':sum(map(len,cksha.values())),
            'manifest_entries':len(manifest(out))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--checkpoints',required=True,type=Path)
    ap.add_argument('--historical',required=True,type=Path)
    ap.add_argument('--verify',action='store_true');ap.add_argument('--verify-sample',action='store_true');ap.add_argument('--resume',action='store_true'); args=ap.parse_args()
    result=(verify(args.out,args.checkpoints,args.historical,args.verify_sample) if args.verify or args.verify_sample else
            new_evidence(args.checkpoints,args.historical,args.out,args.resume))
    print(json.dumps(result if args.verify or args.verify_sample else {'summary':result['summary'],
                            'elapsed_s':result['elapsed_s'],
                            'peak_rss_kib':result['peak_rss_kib']},sort_keys=True))
if __name__=='__main__':main()
