"""Source-verified 128-turn pretrained LFM350 trajectory comparison; no invented labels.

Research only. Inputs: three immutable *historical* CyberSand Actions ZIPs.
Does not execute model weights or validate native CyberSand host.
"""
import argparse
import collections
import csv
import hashlib
import html
import itertools
import json
import math
from pathlib import Path
import zipfile

ARCHIVES = {
    'stock': ('0eed5828082862d74c9dd796b7a45a6110f2eb02326a6a7d5f508e91501072cd', None, 14248333879),
    'gp350006': ('88dfeee5360ae16d03de99d6bdfb2dbe79a819613af215373f6dcfb9fdcf0ae6', '436159bd6508063221d31d46f057b5fba7ee5b6f05b27edf003c54e76863898c', 14244806761),
    'gp350010': ('fa6296ad600176b366b367965b86118fc432a92a63f55c8e0b5a44c488258920', '047b2bacb61f6f04fb359e4d9839f6c198a17ba7db6dfb451d88e48846c66430', 14245172930),
}
MODEL_REVISION = '9921cd4d82c0e3986887aa4b376f9446729ef8a3'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def metrics(positions, decisions):
    pts = [tuple(p) for p in positions]
    steps = [(b[0]-a[0], b[1]-a[1]) for a,b in zip(pts, pts[1:])]
    unit = [s for s in steps if abs(s[0])+abs(s[1]) == 1]
    runs=[]
    for direction, values in itertools.groupby(unit):
        runs.append(len(list(values)))
    turns = sum(a[0]*b[0]+a[1]*b[1] == 0 for a,b in zip(unit,unit[1:]))
    edges = [tuple(sorted((a,b))) for a,b in zip(pts,pts[1:]) if a != b]
    unique_edges = set(edges)
    vertices = set(pts)
    degree=collections.Counter()
    for a,b in unique_edges:
        degree[a]+=1; degree[b]+=1
    actions = collections.Counter(d['applied'] for d in decisions)
    return {
        'ticks':len(decisions), 'distinct_cells':len(vertices),
        'turn90_rate':round(turns/max(1,len(unit)-1),6),
        'mean_straight_run':round(sum(runs)/max(1,len(runs)),6),
        'longest_straight':max(runs,default=0),
        'straight_run_hist':dict(sorted(collections.Counter(runs).items())),
        'two_step_reversals':sum(pts[i]==pts[i-2] for i in range(2,len(pts))),
        'repeated_edge_rate':round((len(edges)-len(unique_edges))/max(1,len(edges)),6),
        'spatial_span':[max(x for x,y in pts)-min(x for x,y in pts),max(y for x,y in pts)-min(y for x,y in pts)],
        'net_displacement':round(math.dist(pts[0],pts[-1]),6),
        'graph_junctions':sum(n>=3 for n in degree.values()),
        'graph_cycle_rank':max(0,len(unique_edges)-len(vertices)+1),
        'branch_actions':actions['BRANCH'], 'retrace_actions':actions['RETRACE'],
        'stop_actions':actions['STOP'], 'terminated_by_stop':bool(actions['STOP']),
    }


def load(arm, path):
    expected, adapter, artifact = ARCHIVES[arm]
    data=Path(path).read_bytes()
    if sha256(data)!=expected: raise ValueError(f'{arm} original ZIP SHA256 mismatch')
    records={}
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None: raise ValueError('corrupt ZIP member')
        root='original-experiment/'
        summary=json.loads(z.read(root+'summary.json'))
        manifest=json.loads(z.read(root+'full-model-manifest.json'))
        if (summary.get('adapter_sha256')!=adapter or summary.get('revision')!=MODEL_REVISION
            or manifest['revision']!=MODEL_REVISION or summary.get('steps_per_episode')!=128
            or summary.get('model_decisions')!=1536 or summary.get('native') is not False
            or summary.get('synthetic_W0_only') is not True):
            raise ValueError('wrong adapter, root model, run size or host attribution')
        expected_png={e['png']:e['png_sha256'] for e in summary['episodes']}
        for name in z.namelist():
            if not name.startswith(root+arm+'-s') or not name.endswith('.json'): continue
            blob=z.read(name); obj=json.loads(blob)
            r=obj['record']; ds=obj['decisions']; es=r['events']; ps=r['trajectory']
            key=(r['size'],r['seed'],r['family'])
            if key in records or len(ds)!=128 or len(es)!=128 or len(ps)!=129:
                raise ValueError('duplicate or incomplete episode')
            if r['engine']!='python-stdlib-baseline-not-native':
                raise ValueError('unexpected host attribution')
            for i,(d,e) in enumerate(zip(ds,es)):
                obs=d['obs']
                if (d['applied'] not in obs['legal'] or d['applied']!=e['applied']
                    or obs!=e['observation'] or list(e['position_after'])!=list(ps[i+1])
                    or [obs['x'],obs['y']]!=list(ps[i])
                    or len(d['six_logits'])!=6 or not all(math.isfinite(x) for x in d['six_logits'])
                    or obs['tick']!=i):
                    raise ValueError(f'{arm} invalid original action / observation at {name}:{i}')
            png=name[:-5]+'.png'
            if png not in z.namelist(): raise ValueError(f'missing source PNG {png}')
            if sha256(z.read(png))!=expected_png.get(png.removeprefix(root)):
                raise ValueError(f'original PNG does not match producer summary: {png}')
            records[key]={
                'stage':arm, 'world':key, 'positions':ps, 'decisions':ds,
                'source_json_sha256':sha256(blob), 'source_png_sha256':sha256(z.read(png)),
                'source_json':name, 'source_artifact_id':artifact,
                'metrics':metrics(ps,ds),
                'model_revision':MODEL_REVISION, 'adapter_sha256':adapter,
                'host':'synthetic W0; not native CyberSand',
            }
    if len(records)!=12: raise ValueError(f'expected 12 source episodes for {arm}, got {len(records)}')
    return records


def svg_page(items, codes):
    # Two matched worlds per A4 landscape page. Display only blind A/B/C codes.
    width,height=1188,840
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="297mm" height="210mm" viewBox="0 0 {width} {height}">',
         '<rect width="1188" height="840" fill="white"/>',
         '<text x="40" y="45" font-size="24" font-family="sans-serif">DRIV3R — ORIGINAL LFM350 · blind paired trajectories</text>',
         '<text x="40" y="68" font-size="13" font-family="sans-serif">128 host-applied synthetic W0 choices per path · line positions are original receipt data · no owner ratings</text>']
    for row,(world,stages) in enumerate(items):
        y0=92+row*370
        label=codes[world]
        out.append(f'<text x="32" y="{y0+17}" font-family="sans-serif" font-size="19">{html.escape(label)}</text>')
        for col,(code,obj) in enumerate(stages):
            x0=34+col*384; py=y0+35
            out.append(f'<rect x="{x0}" y="{py}" width="362" height="307" fill="white" stroke="#bdbdbd"/>')
            out.append(f'<text x="{x0+12}" y="{py+24}" font-family="sans-serif" font-size="17">{code}</text>')
            pts=obj['positions']; size=world[0]
            minx,miny=min(p[0] for p in pts),min(p[1] for p in pts)
            maxx,maxy=max(p[0] for p in pts),max(p[1] for p in pts)
            scale=min(295/max(1,maxx-minx+2),245/max(1,maxy-miny+2))
            def xy(p): return f'{x0+33+(p[0]-minx+1)*scale:.2f},{py+42+(p[1]-miny+1)*scale:.2f}'
            points=' '.join(xy(p) for p in pts)
            out.append(f'<polyline points="{points}" stroke="black" stroke-width="2.2" fill="none" stroke-linejoin="round" stroke-linecap="round"/>')
            for p,fill in [(pts[0],'#888'),(pts[-1],'#222')]:
                px,pyx=xy(p).split(',');out.append(f'<circle cx="{px}" cy="{pyx}" r="4" fill="{fill}"/>')
    out.append('</svg>')
    return '\n'.join(out)+'\n'


def main(archives,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    cohorts={k:load(k,v) for k,v in archives.items()}
    worlds=sorted(cohorts['stock'])
    if not all(set(v)==set(worlds) for v in cohorts.values()):
        raise ValueError('source worlds not matched')
    # stable per-world blind permutations; no source ID in gallery or ballot
    panels={}; reveal=[]
    for i,world in enumerate(worlds):
        names=sorted(cohorts,key=lambda a:sha256((str(world)+a).encode()))
        panels[world]=[(chr(65+j),cohorts[stage][world]) for j,stage in enumerate(names)]
        for code,obj in panels[world]: reveal.append({'world_id':f'W{i+1:02d}','blind_arm':code,'source_stage':obj['stage'],'size':world[0],'seed':world[1],'family':world[2], 'source_json_sha256':obj['source_json_sha256'],'source_png_sha256':obj['source_png_sha256'],'artifact_id':obj['source_artifact_id']})
    pairs=[];ballot=[]
    for i,world in enumerate(worlds):
        for (aa,a),(bb,b) in itertools.combinations(panels[world],2):
            pair_id=f'W{i+1:02d}-{aa}{bb}'
            equal=a['positions']==b['positions']
            prefix=next((j for j,(p,q) in enumerate(zip(a['positions'],b['positions'])) if p!=q),len(a['positions']))
            pairs.append({'pair_id':pair_id,'world_id':f'W{i+1:02d}', 'arm_a':aa,'arm_b':bb,'identical_positions':equal,'common_position_prefix':prefix,'same_horizon':True,'same_world':True, 'source_disjoint_holdout':False,'owner_human_ratings':0})
            for niche in ['MEANDERER','CURLER','BRANCHER','CHAOTIC_ARTIST']:
                ballot.append({'pair_id':pair_id,'niche':niche,'choice_A_B_TIE_SKIP':'','rater_kind':'owner/human','notes':''})
    for j in range(0,len(worlds),2):
        name=f'P21_BLIND_A4_{j//2+1:02d}.svg'
        (out/name).write_text(svg_page([(w,panels[w]) for w in worlds[j:j+2]], {w:f'World W{idx+1:02d}' for idx,w in enumerate(worlds)}))
    def writecsv(name,rows):
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    writecsv('P21_REVEAL.csv',reveal);writecsv('P21_PAIRS.csv',pairs);writecsv('P21_BLANK_OWNER_BALLOT.csv',ballot)
    catalog=[]
    for i,w in enumerate(worlds):
        for code,obj in panels[w]:
            catalog.append({'world_id':f'W{i+1:02d}','blind_arm':code,'source_stage':obj['stage'],**obj['metrics'],'source_json_sha256':obj['source_json_sha256'],'source_png_sha256':obj['source_png_sha256'],'source_artifact_id':obj['source_artifact_id']})
    (out/'P21_CATALOG.json').write_text(json.dumps(catalog,indent=2,sort_keys=True)+'\n')
    stats={k:{metric:round(sum(o['metrics'][metric] for o in episodes.values())/12,5) for metric in ['distinct_cells','turn90_rate','longest_straight','two_step_reversals','repeated_edge_rate','graph_junctions','graph_cycle_rank','branch_actions','retrace_actions','stop_actions']} for k,episodes in cohorts.items()}
    result={'schema':'driv3r.preference_eval.p21_source_verified_lfm350/v1','model_revision':MODEL_REVISION,'sources':{k:{'zip_sha256':ARCHIVES[k][0],'adapter_sha256':ARCHIVES[k][1],'artifact_id':ARCHIVES[k][2]} for k in cohorts},'worlds':12,'independent_seeds':2,'source_episodes':36,'total_model_decisions':4608,'host_illegal_actions':0,'model_forward_this_run':0,'optimizer_updates_this_run':0,'matched_pairs':36,'identical_pairs':sum(p['identical_positions'] for p in pairs),'nonidentical_pairs':sum(not p['identical_positions'] for p in pairs),'distinct_pairs_by_stage':{f'{a}/{b}':sum(cohorts[a][w]['positions']!=cohorts[b][w]['positions'] for w in worlds) for a,b in itertools.combinations(cohorts,2)},'mean_by_stage':stats,'owner_human_ratings':0,'blind_svg_sheets':6,'qualification':'historical original model-applied synthetic W0 source archive audit; not fresh SLM inference/native game/independent heldout evaluation'}
    (out/'P21_SUMMARY.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    checks={f.name:sha256(f.read_bytes()) for f in sorted(out.iterdir()) if f.is_file()}
    (out/'SHA256SUMS.json').write_text(json.dumps(checks,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--stock',required=True);ap.add_argument('--gp006',required=True);ap.add_argument('--gp010',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();main({'stock':a.stock,'gp350006':a.gp006,'gp350010':a.gp010},a.out)
