"""P17 secondary-source cross-model descriptors; no invented action or owner labels.

Input is the curated 14-specimen original-position manifest with per-trace
producer artifact IDs and original JSON SHA-256 references. Original action JSON
bytes are NOT embedded here, so legality, STOP and root model choices cannot be
independently reconstructed in this evaluation.
"""
import argparse
import collections
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from itertools import combinations


def sha(b): return hashlib.sha256(b).hexdigest()

def unit(a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    return (dx,dy) if abs(dx)+abs(dy)==1 else None


def geometry(positions):
    p=[tuple(x) for x in positions]
    dirs=[unit(a,b) for a,b in zip(p,p[1:])]
    edges=[tuple(sorted((a,b))) for a,b in zip(p,p[1:]) if unit(a,b)]
    hist={str(k):0 for k in ('1','2','3-4','5-8','9+')}
    runs=[];n=0; signed=[]; t90=0; rev=0
    for i,d in enumerate(dirs):
        last=dirs[i-1] if i else None
        if d is None:
            if n:runs.append(n)
            n=0
        elif d==last:n+=1
        else:
            if n:runs.append(n)
            n=1
            if last:
                dot=d[0]*last[0]+d[1]*last[1]
                cross=last[0]*d[1]-last[1]*d[0]
                if dot==0: t90+=1; signed.append((i,1 if cross>0 else -1))
                if dot==-1: rev+=1
    if n:runs.append(n)
    for r in runs:hist['1' if r==1 else '2' if r==2 else '3-4' if r<5 else '5-8' if r<9 else '9+']+=1
    neighbors=collections.defaultdict(set)
    for a,b in set(edges):neighbors[a].add(b);neighbors[b].add(a)
    # Graph-cycle rank for a connected path with separate teleport components allowed.
    vertices=set(p);par={x:x for x in vertices}
    def root(v):
        while par[v]!=v:par[v]=par[par[v]];v=par[v]
        return v
    for a,b in set(edges):par[root(a)]=root(b)
    cycles=max(0,len(set(edges))-len(vertices)+len({root(v) for v in vertices}))
    xs=[x for x,y in p];ys=[y for x,y in p]
    return {'turn90_rate':round(t90/max(1,len([d for d in dirs if d])-1),5),
        'run_length_hist':hist,'longest_straight':max(runs,default=0),
        'reversal2':sum(p[i]==p[i+2] and bool(dirs[i]) and bool(dirs[i+1]) for i in range(len(p)-2)),
        'repeated_edge_rate':round((len(edges)-len(set(edges)))/max(1,len(edges)),5),
        'xspan':max(xs)-min(xs)+1,'yspan':max(ys)-min(ys)+1,
        'net_displacement':round(math.dist(p[0],p[-1]),3),'unique_cells':len(vertices),
        'junction_motifs':sum(len(n)>=3 for n in neighbors.values()),'cycles':cycles,
        'hooks':sum(j-i<=10 and s==t for (i,s),(j,t) in zip(signed,signed[1:])),
        'nonlocal_steps':sum(d is None for d in dirs),
        'model_action_legality':None,'early_STOP':None,'branch_actions':None,
        'host_legal_action_count':None}


def descriptor(m,niche):
    if niche=='MEANDERER':return m['turn90_rate']*min(80,m['xspan']+m['yspan'])*(1-m['repeated_edge_rate'])
    if niche=='CURLER':return m['hooks']+3*m['cycles']
    if niche=='BRANCHER':return m['junction_motifs']+3*m['cycles']
    return m['turn90_rate']*min(64,m['unique_cells'])*(1-m['repeated_edge_rate'])


def viable_shape(m):
    # Observed-shape gate only: it says nothing about missing legality receipts.
    return m['unique_cells']>=8 and m['reversal2']<=max(4,m['unique_cells']//3) and m['repeated_edge_rate']<=0.25


def svg(rows,niche):
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 210 297">',
        '<rect width="210" height="297" fill="white"/>',
        f'<text x="12" y="14" font-family="sans-serif" font-size="6">{html.escape(niche)} · blind historical paths</text>',
        '<text x="12" y="21" font-family="sans-serif" font-size="3">Original-position secondary curation · identities deliberately withheld</text>']
    for i,r in enumerate(rows):
        x=12+99*(i%2);y=29+85*(i//2)
        pos=r['positions'];px=[a for a,b in pos];py=[b for a,b in pos]
        lowx,lowy=min(px),min(py)
        scale=min(80/max(1,max(px)-lowx+1),68/max(1,max(py)-lowy+1))
        pts=' '.join(f'{x+5+(a-lowx)*scale:.2f},{y+7+(b-lowy)*scale:.2f}' for a,b in pos)
        out += [f'<rect x="{x}" y="{y}" width="90" height="77" fill="none" stroke="#ddd" stroke-width=".3"/>',
                f'<text x="{x+3}" y="{y+5}" font-family="monospace" font-size="3.5">SOURCE {i+1:02d}</text>',
                f'<polyline points="{pts}" fill="none" stroke="#222" stroke-width=".7" stroke-linejoin="round"/>',
                f'<circle cx="{x+5+(px[0]-lowx)*scale:.2f}" cy="{y+7+(py[0]-lowy)*scale:.2f}" r="1" fill="#35745e"/>',
                f'<circle cx="{x+5+(px[-1]-lowx)*scale:.2f}" cy="{y+7+(py[-1]-lowy)*scale:.2f}" r="1" fill="#ad4d42"/>']
    out.append('<text x="12" y="290" font-family="sans-serif" font-size="2.8">No owner votes · host/legal receipts not embedded · paths auto-fitted, not size-comparable</text></svg>')
    return '\n'.join(out)+'\n'


def run(manifest_path,outdir):
    raw=Path(manifest_path).read_bytes(); obj=json.loads(raw)
    rows=[]
    for x in obj['specimens']:
        assert len(x['positions'])==x['decisions']+1
        assert len(x['pen_down_segments'])==x['decisions']
        assert len(x['source']['json_sha256'])==64
        d=geometry(x['positions'])
        rows.append({'name':x['name'],'group':x['group'],'world':x['world'],
            'decisions':x['decisions'],'source':x['source'],'positions':x['positions'],
            'positions_sha256':sha(json.dumps(x['positions'],separators=(',',':')).encode()),
            'geometry':d,'shape_gate':viable_shape(d)})
    assert len(rows)==14 and len({r['name'] for r in rows})==14
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    catalog={'schema':'driv3r.preference.p17.crossmodel_curated/1',
        'curation_file_sha256':sha(raw),'source_integrity':'secondary, original JSON bytes not present in this input',
        'model_generated':sum(r['group']!='native' for r in rows),
        'native_references':sum(r['group']=='native' for r in rows),
        'human_ratings':None,'cross_model_heldout_ranker':'NOT_TRAINED_INSUFFICIENT_INDEPENDENT_MATCHED_WORLDS',
        'niches':{},'entries':[{k:v for k,v in r.items() if k!='positions'} for r in rows]}
    for niche in ('MEANDERER','CURLER','BRANCHER','CHAOTIC_ARTIST'):
        ranked=sorted([r for r in rows if r['shape_gate']],key=lambda r:(descriptor(r['geometry'],niche),r['positions_sha256']),reverse=True)
        chosen=[];groups=collections.Counter()
        for r in ranked:
            if groups[r['group']]>=2:continue
            chosen.append(r);groups[r['group']]+=1
            if len(chosen)>=6:break
        (outdir/f'P17-cross-{niche}-blind-A4.svg').write_text(svg(chosen,niche))
        catalog['niches'][niche]=[{'anonymous_id':f'{niche}-{i+1:02d}',
            'name_reveal_after_blind_review':r['name'],'positions_sha256':r['positions_sha256'],
            'geometry_descriptor':round(descriptor(r['geometry'],niche),4)} for i,r in enumerate(chosen)]
    # Same source world AND same horizon, otherwise comparisons are invalid.
    pairs=[]
    for a,b in combinations(rows,2):
        if (a['group'],a['world'],a['decisions'])!=(b['group'],b['world'],b['decisions']):continue
        pairs.append({'a':a['name'],'b':b['name'],'group':a['group'],'world':a['world'],
            'horizon':a['decisions'],'a_json_sha256':a['source']['json_sha256'],
            'b_json_sha256':b['source']['json_sha256'],
            'owner_preference':None,'model_preference':None,'weak_supervision_label':None,
            'descriptor_delta':{n:round(descriptor(a['geometry'],n)-descriptor(b['geometry'],n),4)
                                  for n in ('MEANDERER','CURLER','BRANCHER','CHAOTIC_ARTIST')}})
    catalog['matched_pairs_without_preference_labels']=pairs
    (outdir/'P17-cross-model-catalogue.json').write_text(json.dumps(catalog,indent=2,sort_keys=True)+'\n')
    with (outdir/'P17-cross-model-descriptors.csv').open('w',newline='') as f:
        keys=['name','group','world','decisions','positions_sha256','turn90_rate','longest_straight',
              'reversal2','repeated_edge_rate','xspan','yspan','net_displacement','unique_cells',
              'junction_motifs','cycles','hooks','nonlocal_steps','shape_gate']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:r.get(k,r['geometry'].get(k,'')) for k in keys})
    return catalog

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(); c=run(a.manifest,a.output)
    print(len(c['entries']),'source curated specimens;',len(c['matched_pairs_without_preference_labels']),'valid matched pairs, 0 human labels')
