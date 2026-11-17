"""P17 real W16 trajectory niche audit; synthetic W0 evidence, NOT human ratings.

Pure-standard-library source. Reads only archived original intact W16 trace JSONs.
Never infers missing model trajectories from aggregate ledgers or paintings.
"""
import argparse
import collections
import csv
import hashlib
import html
import json
from pathlib import Path
import zipfile

NICHE = ('MEANDERER', 'CURLER', 'BRANCHER', 'CHAOTIC_ARTIST')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unit(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) + abs(dy) == 1:
        return dx, dy
    return None


def analyze(doc):
    steps = doc['events']
    path = list(map(tuple, doc['trajectory']))
    assert len(path) == len(steps) + 1, 'trajectory/event count mismatch'
    assert doc['steps_executed'] == len(steps), 'step count mismatch'
    assert all(len(p) == 2 and all(isinstance(q, int) and 0 <= q < doc['size'] for q in p) for p in path)
    illegal = 0
    for i, e in enumerate(steps):
        assert tuple(e['position_after']) == path[i+1], 'position receipt mismatch'
        assert (e['observation']['x'], e['observation']['y']) == path[i], 'observation/trajectory mismatch'
        assert e['applied'] in e['observation']['legal'], 'host illegal applied action'
        assert e['observation']['tick'] == i, 'tick mismatch'
        illegal += int(not e.get('raw_legal', False))
    moves = [unit(a,b) for a,b in zip(path, path[1:])]
    valid = [d for d in moves if d is not None]
    turns, reversals, left, right, run_lengths = 0, 0, 0, 0, []
    run = 0
    for i,d in enumerate(moves):
        if d is None:
            if run: run_lengths.append(run)
            run = 0
            continue
        if not run: run = 1
        else:
            prior = moves[i-1]
            if prior == d: run += 1
            else:
                run_lengths.append(run)
                run = 1
                if prior:
                    dot = prior[0]*d[0] + prior[1]*d[1]
                    cross = prior[0]*d[1] - prior[1]*d[0]
                    if dot == 0:
                        turns += 1
                        left += int(cross > 0)
                        right += int(cross < 0)
                    if dot == -1: reversals += 1
    if run: run_lengths.append(run)
    undirected = [tuple(sorted((a,b))) for a,b in zip(path,path[1:]) if unit(a,b)]
    repeats = len(undirected) - len(set(undirected))
    uniq = set(path)
    xs,ys = [p[0] for p in path], [p[1] for p in path]
    neighbors = collections.defaultdict(set)
    for a,b in set(undirected): neighbors[a].add(b);neighbors[b].add(a)
    junctions = sum(len(n)>=3 for n in neighbors.values())
    # Graph cycle rank over visited vertices and actual unit edges; not painted-area loops.
    parents={p:p for p in uniq}
    def root(x):
        while parents[x]!=x:
            parents[x]=parents[parents[x]];x=parents[x]
        return x
    for a,b in set(undirected): parents[root(a)]=root(b)
    cycles=max(0,len(set(undirected))-len(uniq)+len({root(p) for p in uniq}))
    # Compact same-handed corner sequences (hooks/curves), not a validated spiral detector.
    signed=[]
    for i in range(1,len(moves)):
        a,b=moves[i-1],moves[i]
        if a and b and a[0]*b[0]+a[1]*b[1]==0:
            signed.append((i, 1 if a[0]*b[1]-a[1]*b[0]>0 else -1))
    hooks=sum(j-i<=10 and s==t for (i,s),(j,t) in zip(signed,signed[1:]))
    early_stop=any(e['applied']=='STOP' and i < int(.75*doc['step_budget']) for i,e in enumerate(steps))
    hist={k:0 for k in ('1','2','3-4','5-8','9+')}
    for n in run_lengths:
        hist['1' if n==1 else '2' if n==2 else '3-4' if n<=4 else '5-8' if n<=8 else '9+']+=1
    return dict(turn90_rate=round(turns/max(1,len(valid)-1),5), straight_hist=hist,
        longest_straight=max(run_lengths,default=0), reversal2=sum(bool(path[i]==path[i+2] and moves[i] and moves[i+1]) for i in range(len(path)-2)),
        directional_reversals=reversals, repeated_edges=repeats, repeated_edge_rate=round(repeats/max(1,len(undirected)),5),
        xspan=max(xs)-min(xs)+1,yspan=max(ys)-min(ys)+1,net_displacement=round(((xs[-1]-xs[0])**2+(ys[-1]-ys[0])**2)**0.5,3),
        unique_cells=len(uniq), junction_motifs=junctions,cycles=cycles,hooks=hooks,
        branch_actions=sum(e['applied']=='BRANCH' for e in steps), retrace_actions=sum(e['applied']=='RETRACE' for e in steps),
        stop_actions=sum(e['applied']=='STOP' for e in steps), early_stop=early_stop,
        nonterminal_steps=sum(e['applied']!='STOP' for e in steps),host_illegal_actions=0,raw_illegal_choices=illegal,
        transport_steps=sum(d is None for d in moves),left_turns=left,right_turns=right,
        unit_moves=len(valid), step_budget=doc['step_budget'])


def load_archive(archive):
    kind='P16' if 'P16_' in archive.name else 'P10' if 'P10_' in archive.name else None
    assert kind, 'Only authentic P16/P10 sources accepted in this bounded P17 audit'
    archive_hash=sha(archive.read_bytes())
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            if kind=='P16':
                ok=name.startswith('P16/independent_outputs/') and name.endswith('-original_late.json')
            else:
                ok=name.startswith('new-rollouts/') and name.endswith('-late-intact.json')
            if not ok: continue
            raw=z.read(name); doc=json.loads(raw)
            assert doc['contract']=='cybersand.worm.observation-action/0-experimental'
            m=analyze(doc)
            yield dict(source=kind,source_zip_sha=archive_hash,entry=name,trace_sha=sha(raw),
                family=doc['family'], seed=doc['seed'], model_class='frozen W16 142-feature auxiliary PyTorch MLP',
                decoder='original late-STOP intact', metrics=m, path=doc['trajectory'])


def viable(m):
    return m['host_illegal_actions']==0 and m['raw_illegal_choices']==0 and m['unique_cells']>=32 and not m['early_stop'] and m['reversal2']<=16 and m['repeated_edge_rate']<=0.20


def axis_score(m, niche):
    span=m['xspan']+m['yspan']
    if niche=='MEANDERER': return (m['turn90_rate']*min(span,80)*(1-m['repeated_edge_rate']),m['net_displacement'])
    if niche=='CURLER': return (m['hooks']+3*m['cycles'],m['hooks'])
    if niche=='BRANCHER': return (m['junction_motifs']+2*m['branch_actions'],m['cycles'])
    return (m['turn90_rate']*min(60,m['unique_cells'])*(1-m['repeated_edge_rate']),m['hooks']+m['junction_motifs'])


def choose(rows,niche,n=6):
    # Avoid six replicates of one seed/family; score is a display descriptor, not quality.
    elig=[r for r in rows if viable(r['metrics'])]
    ranked=sorted(elig,key=lambda r:(axis_score(r['metrics'],niche),r['trace_sha']),reverse=True)
    chosen=[]; seeds=set(); families=collections.Counter()
    for r in ranked:
        if r['seed'] in seeds or families[r['family']]>=2: continue
        chosen.append(r); seeds.add(r['seed']); families[r['family']]+=1
        if len(chosen)==n:break
    return chosen


def svg_sheet(rows, niche):
    # A4 portrait; no model/checkpoint/seed/family identity on the blind sheet.
    esc=html.escape
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 210 297">',
         '<rect width="210" height="297" fill="#fff"/>',
         f'<text x="12" y="15" font-family="sans-serif" font-weight="bold" font-size="6">{esc(niche)} · blind W0 trajectories</text>',
         '<text x="12" y="21" font-family="sans-serif" font-size="3">Anonymous identifiers · original paths only · model labels withheld</text>']
    for k,r in enumerate(rows):
        x=12+(k%2)*99;y=29+(k//2)*85
        coords=r['path']; px=[p[0] for p in coords];py=[p[1] for p in coords];
        lo_x,lo_y=min(px),min(py); scale=min(80/max(1,max(px)-lo_x+1),68/max(1,max(py)-lo_y+1))
        pts=' '.join(f'{x+5+(a-lo_x)*scale:.2f},{y+7+(b-lo_y)*scale:.2f}' for a,b in coords)
        out += [f'<rect x="{x}" y="{y}" width="90" height="77" fill="none" stroke="#ddd" stroke-width="0.3"/>',
                f'<text x="{x+3}" y="{y+5}" font-family="monospace" font-size="3.5">SPECIMEN {k+1:02d}</text>',
                f'<polyline points="{pts}" fill="none" stroke="#20252a" stroke-width="0.65" stroke-linejoin="round" stroke-linecap="round"/>',
                f'<circle cx="{x+5+(px[0]-lo_x)*scale:.2f}" cy="{y+7+(py[0]-lo_y)*scale:.2f}" r="1.0" fill="#4a8f75"/>',
                f'<circle cx="{x+5+(px[-1]-lo_x)*scale:.2f}" cy="{y+7+(py[-1]-lo_y)*scale:.2f}" r="0.9" fill="#b25a55"/>']
    out.append('<text x="12" y="290" font-family="sans-serif" font-size="2.8">Green start · red end · not native CyberSand · no human aesthetic judgements</text></svg>')
    return '\n'.join(out)+'\n'


def run(archives,outdir):
    rows=[r for a in archives for r in load_archive(Path(a))]
    original_count=len(rows)
    # Multiple hypothetical P16 masks leave original_late unchanged: do not pseudoreplicate.
    dedup={}
    for r in rows:
        if r['trace_sha'] in dedup:
            old=dedup[r['trace_sha']]
            assert (old['source'],old['seed'],old['family'])==(r['source'],r['seed'],r['family']), 'cross-world duplicate collision'
        else: dedup[r['trace_sha']]=r
    rows=list(dedup.values())
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    for r in rows: r['eligible']=viable(r['metrics'])
    catalogue={'schema':'driv3r.preference.p17.trajectory_niches/1',
        'source_boundary':'P10/P16 actual frozen W16 auxiliary model, synthetic W0, not independent root SLMs or human ratings',
        'source_disjoint_cross_model_evaluation':'NOT POSSIBLE FROM PRESENT INPUTS',
        'cohorts':dict(collections.Counter(r['source'] for r in rows)),
        'original_execution_receipts':original_count,'duplicate_original_receipts':original_count-len(rows),
        'episodes':len(rows),'eligible':sum(r['eligible'] for r in rows),'early_stop':sum(r['metrics']['early_stop'] for r in rows),
        'niches':{},'entries':[]}
    for niche in NICHE:
        selected=choose(rows,niche)
        (outdir/f'P17-{niche}-blind-A4.svg').write_text(svg_sheet(selected,niche),encoding='utf-8')
        catalogue['niches'][niche]=[{'id':f'{niche}-{i+1:02d}','trace_sha':r['trace_sha'],
             'source':r['source'],'entry':r['entry'],'primary_descriptor':round(axis_score(r['metrics'],niche)[0],4)}
              for i,r in enumerate(selected)]
    for r in rows:
        catalogue['entries'].append({k:v for k,v in r.items() if k!='path'})
    (outdir/'P17-catalogue.json').write_text(json.dumps(catalogue,indent=2,sort_keys=True)+'\n')
    with (outdir/'P17-descriptors.csv').open('w',newline='') as fd:
        cols=['source','family','seed','trace_sha','eligible','turn90_rate','longest_straight','reversal2',
              'repeated_edge_rate','xspan','yspan','net_displacement','unique_cells','junction_motifs',
              'cycles','hooks','branch_actions','retrace_actions','stop_actions','early_stop','raw_illegal_choices','transport_steps']
        wr=csv.DictWriter(fd,fieldnames=cols);wr.writeheader()
        for r in rows:
            wr.writerow({k:r.get(k,r['metrics'].get(k,'')) for k in cols})
    return catalogue


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--zip',action='append',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();c=run(args.zip,args.output)
    print(f"Original W16 episodes: {c['episodes']}; viable: {c['eligible']}; per-niche selections: "+str({k:len(v) for k,v in c['niches'].items()}))
