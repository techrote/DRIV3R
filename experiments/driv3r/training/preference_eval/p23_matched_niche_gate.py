"""P23 original-source-only paired four-niche gate. No generated preferences or SLM inference.

Inputs must be independently SHA-256-verified by P18/P21 auditors. 128-, 36-,
and 16-decision cohorts are not pooled for model-preference estimates.
"""
import argparse
import collections
import csv
import hashlib
import html
import itertools
import json
from pathlib import Path

from p18_reverify_ballots import paired as historical_pairs, verify as verify_historical
from p21_real_lfm_bestiary import load as load_lfm

NICHES = ('MEANDERER','CURLER','BRANCHER','CHAOTIC_ARTIST')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def traits(positions):
    p = [tuple(a) for a in positions]
    edges = [tuple(sorted((a,b))) for a,b in zip(p,p[1:]) if a!=b]
    e = set(edges)
    degree = collections.Counter()
    for a,b in e:
        degree[a]+=1; degree[b]+=1
    moves = [(b[0]-a[0],b[1]-a[1]) for a,b in zip(p,p[1:])]
    unit = [v for v in moves if abs(v[0])+abs(v[1])==1]
    runs = [len(list(g)) for _,g in itertools.groupby(unit)]
    turns = sum(a[0]*b[0]+a[1]*b[1]==0 for a,b in zip(unit,unit[1:]))
    unique = len(set(p)); horizon=len(p)-1
    d = {'horizon':horizon, 'unique_cells':unique, 'unique_per_tick':round(unique/max(1,horizon),6),
         'turn90_rate':round(turns/max(1,len(unit)-1),6),
         'longest_straight':max(runs,default=0),
         'straight_run_hist':dict(sorted(collections.Counter(runs).items())),
         'two_step_reversals':sum(p[i]==p[i-2] for i in range(2,len(p))),
         'repeated_edge_rate':round((len(edges)-len(e))/max(1,len(edges)),6),
         'junctions':sum(v>=3 for v in degree.values()),
         'cycle_rank':max(0,len(e)-unique+1),
         'span_x':max(x for x,y in p)-min(x for x,y in p),
         'span_y':max(y for x,y in p)-min(y for x,y in p),
         'net_manhattan':abs(p[-1][0]-p[0][0])+abs(p[-1][1]-p[0][1])}
    flags=[]
    if horizon>=64 and unique<=4: flags.append('tiny_stationary_loop')
    if d['two_step_reversals']/max(1,horizon)>=.10: flags.append('two_step_oscillation')
    if d['repeated_edge_rate']>.25 and d['unique_per_tick']<.75: flags.append('high_turn_confinement')
    d['screening_flags']=flags  # structural screening, NOT a human aesthetic score
    return d


def svg(pairs):
    lines=['<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 210 297">',
           '<rect width="210" height="297" fill="white"/>',
           '<text x="10" y="12" font-family="sans-serif" font-size="6">Worm trajectory comparison — blind A4</text>',
           '<text x="10" y="18" font-family="sans-serif" font-size="3">Original applied traces · four distinct niches · no labels or derived scores</text>']
    for idx,pair in enumerate(pairs):
        y=25+idx*130
        a,b=pair['arms'];pts=a['positions']+b['positions']
        lo_x=min(p[0] for p in pts);lo_y=min(p[1] for p in pts)
        span_x=max(p[0] for p in pts)-lo_x+1;span_y=max(p[1] for p in pts)-lo_y+1
        scale=min(76/span_x,88/span_y)
        lines.append(f'<text x="10" y="{y}" font-family="monospace" font-size="4">{pair["blind_id"]}</text>')
        for col,arm in enumerate((a,b)):
            x=10+col*99
            lines += [f'<rect x="{x}" y="{y+5}" width="91" height="111" fill="white" stroke="#c7c7c7" stroke-width=".35"/>',
                      f'<text x="{x+3}" y="{y+12}" font-size="4" font-family="sans-serif">{"A" if col==0 else "B"}</text>']
            path=arm['positions'];ink=arm.get('ink',[True]*(len(path)-1))
            for j,(start,end) in enumerate(zip(path,path[1:])):
                if start==end: continue
                x1=x+8+(start[0]-lo_x)*scale;y1=y+18+(start[1]-lo_y)*scale
                x2=x+8+(end[0]-lo_x)*scale;y2=y+18+(end[1]-lo_y)*scale
                color='#252525' if ink[j] else '#bdbdbd'
                dash='' if ink[j] else ' stroke-dasharray="1 1"'
                lines.append(f'<path d="M {x1:.2f} {y1:.2f} L {x2:.2f} {y2:.2f}" stroke="{color}" stroke-width=".65" fill="none"{dash}/>')
            for pt,color in ((path[0],'#256d4d'),(path[-1],'#a04747')):
                xx=x+8+(pt[0]-lo_x)*scale;yy=y+18+(pt[1]-lo_y)*scale
                lines.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="1.2" fill="{color}"/>')
    lines += ['<text x="10" y="291" font-size="3" font-family="sans-serif">Equal scale and orientation only within each pair · gray = pen lifted</text>',
              '<text x="10" y="295" font-size="3" font-family="sans-serif">Green start; red end. Same source world and action horizon within each pair.</text>', '</svg>']
    return '\n'.join(lines)+'\n'


def run(stock,gp006,gp010,manifest,old_sources,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    original={'stock':load_lfm('stock',stock),'gp350006':load_lfm('gp350006',gp006),'gp350010':load_lfm('gp350010',gp010)}
    source_rows=verify_historical(manifest,old_sources)
    all_candidates=[];p={};checked=0
    for arm,records in original.items():
        for world,src in sorted(records.items()):
            key=sha(src['source_json_sha256'].encode())[:14]
            m=traits(src['positions']);checked+=len(src['decisions'])
            item={'candidate_id':key,'source_kind':'real_pretrained_LFM350_original_applied_synthetic_W0',
                  'stage':arm,'world':world,'source_artifact':src['source_artifact_id'],
                  'original_json_sha256':src['source_json_sha256'], 'metrics':m}
            all_candidates.append(item)
            p[('gp',arm,world)]={'positions':src['positions'],'sha':src['source_json_sha256'],'stage':arm,'host':'W0','ink':[True]*128,'metrics':m}
    for row in source_rows:
        key=sha(row['source_json_sha256'].encode())[:14]
        m=traits(row['positions']);checked+=row['decisions']
        all_candidates.append({'candidate_id':key,'source_kind':'historical_original_applied_'+row['host'],
            'stage':row['group'],'world':row['world'],'source_artifact':row['source_artifact_id'],
            'original_json_sha256':row['source_json_sha256'],'metrics':m})
    if len(all_candidates)!=50: raise ValueError('expected 50 individually verified original paths')
    pairs=[];same_control=[];changed=[]
    for world in sorted(original['gp350006']):
        a,b=p[('gp','gp350006',world)],p[('gp','gp350010',world)]
        if a['positions']==b['positions']:
            same_control.append((world,a,b))
        else: changed.append((world,a,b))
    if len(changed)!=4 or len(same_control)!=8: raise ValueError('unexpected real LFM source changes')
    for world,a,b in changed:
        pairs.append({'world_block':'gp128/'+str(world), 'group':'gp128', 'arms':(a,b), 'identical':False})
    for pair in historical_pairs(source_rows):
        a,b=pair['first'],pair['second']
        def convert(row):
            return {'positions':row['positions'],'ink':row['ink'],'sha':row['source_json_sha256'],
                'stage':row['name'],'host':row['host'],'metrics':traits(row['positions'])}
        aa,bb=convert(a),convert(b)
        if aa['positions']==bb['positions']: raise ValueError('historical pair wrongly identical')
        pairs.append({'world_block':'hist/'+str(pair['world_block']), 'group':pair['first']['group'],
                      'arms':(aa,bb),'identical':False})
    # Two deliberately identical catch trials, kept hidden from the displayed sheet/ballot.
    for world,a,b in same_control[:2]:
        pairs.append({'world_block':'gp128/'+str(world), 'group':'gp128', 'arms':(a,b), 'identical':True})
    if len(pairs)!=16: raise ValueError('expected 14 true comparisons + 2 attention controls')
    # Deterministic anonymous codes, no stage/family visible on SVG or ballot.
    pairs.sort(key=lambda x: sha((x['world_block']+x['arms'][0]['sha']+x['arms'][1]['sha']).encode()))
    for k,pair in enumerate(pairs):
        pair['blind_id']='X'+sha((str(k)+pair['world_block']).encode())[:10]
        if int(sha(pair['blind_id'].encode())[:2],16)%2:
            pair['arms']=(pair['arms'][1],pair['arms'][0])
    reveal=[];ballot=[]
    for pair in pairs:
        a,b=pair['arms']
        reveal.append({'blind_id':pair['blind_id'],'group':pair['group'],'world_block':pair['world_block'],
                       'horizon':len(a['positions'])-1,'identical_visual_catch':pair['identical'],
                       'A':{'stage':a['stage'],'original_json_sha256':a['sha'],'traits':a['metrics']},
                       'B':{'stage':b['stage'],'original_json_sha256':b['sha'],'traits':b['metrics']}})
        for n in NICHES: ballot.append({'blind_id':pair['blind_id'],'niche':n,'choice_A_B_TIE_SKIP':'','confidence_1_to_5':'','owner_notes':''})
    for j in range(0,16,2):
        (out/f'P23_BLIND_A4_{j//2+1:02d}.svg').write_text(svg(pairs[j:j+2]))
    with (out/'P23_BLANK_OWNER_BALLOT.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=ballot[0].keys());w.writeheader();w.writerows(ballot)
    (out/'P23_REVEAL.json').write_text(json.dumps(reveal,indent=2,sort_keys=True)+'\n')
    (out/'P23_CANDIDATES.json').write_text(json.dumps(all_candidates,indent=2,sort_keys=True)+'\n')
    by_seed={}
    for seed in sorted({w[1] for w in original['stock']}):
        ww=[w for w in original['stock'] if w[1]==seed]
        by_seed[str(seed)]={'worlds':len(ww),
            'gp010_changed_vs_gp006':sum(original['gp350006'][w]['positions']!=original['gp350010'][w]['positions'] for w in ww),
            'gp010_more_turns':sum(traits(original['gp350010'][w]['positions'])['turn90_rate']>traits(original['gp350006'][w]['positions'])['turn90_rate'] for w in ww),
            'gp010_fewer_unique':sum(traits(original['gp350010'][w]['positions'])['unique_cells']<traits(original['gp350006'][w]['positions'])['unique_cells'] for w in ww)}
    neg=[{'world':list(w),'gp006_unique':traits(original['gp350006'][w]['positions'])['unique_cells'],
          'gp010_unique':traits(original['gp350010'][w]['positions'])['unique_cells'],
          'gp010_turn90':traits(original['gp350010'][w]['positions'])['turn90_rate'],
          'gp010_repeated_edge':traits(original['gp350010'][w]['positions'])['repeated_edge_rate']}
         for w in sorted(original['gp350006']) if 'high_turn_confinement' in traits(original['gp350010'][w]['positions'])['screening_flags']]
    summary={'schema':'driv3r.preference_eval.p23_original_crossmodel_niche_gate/1',
             'source_original_trajectories':50,'real_pretrained_LFM350_episodes':36,
             'historical_original_model_W0':12,'historical_native_reference':2,
             'original_model_applied_decisions_verified_by_upstream_auditors':checked,
             'matched_true_nonidentical_pairs':14,'same_horizon_identical_visual_controls':2,
             'nonidentical_full_LFM128_GP006_GP010':4,'full_LFM128_identical_excluded':8,
             'historical_true_nonidentical_pairs':10,
             'distinct_source_world_blocks_in_nonidentical_model_pairs':6,
             'independent_source_seeds_in_LFM128':2,
             'independent_source_world_blocks_native_ref':1,
             'by_seed_GP006_GP010':by_seed,'high_turn_confinement_GP010_worlds':neg,
             'blind_A4_sheets':8,'blank_owner_rating_rows':64,
             'human_owner_labels':0,'machine_aesthetic_labels':0,'new_root_SLM_forwards':0,'new_optimizer_updates':0,
             'qualification':'historical original-source check with SHA and legal menus via P18/P21; local-only; not newly Actions-qualified or native'}
    (out/'P23_RESULTS.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({q.name:sha(q.read_bytes()) for q in sorted(out.iterdir()) if q.is_file()},indent=2,sort_keys=True)+'\n')
    return summary

if __name__=='__main__':
    cli=argparse.ArgumentParser()
    for opt in ('stock','gp006','gp010','manifest','old-sources','out'):cli.add_argument('--'+opt,required=True)
    args=cli.parse_args()
    print(json.dumps(run(args.stock,args.gp006,args.gp010,args.manifest,args.old_sources,args.out),indent=2))
