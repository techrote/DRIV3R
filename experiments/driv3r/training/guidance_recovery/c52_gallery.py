"""Render a verified SVG gallery from live host-executed C52 suffix paths."""
from __future__ import annotations
import argparse, json, collections, html
from pathlib import Path
import numpy as np
import torch
import c52_shortest_step_cue as c52
import c50_continuous_recovery as c50
from safetensors.torch import load_file

COLORS={'C18-A0':'#9c9c9c','C34-RESET':'#008cc3','C40-PAIR':'#e07829',
'C52-ALIGNED':'#327b5c','C52-SHUFFLED':'#bc588c','C52-RANDOM':'#7f65bd'}

def build(root,out):
    root=Path(root);out=Path(out)
    base,opt,prior=c50.c47.c44.prerequisites(root/'prereq/prereq50/previous/prev47/previous/predecessor_C42')
    data=json.loads((out/'C52-online.json').read_text())
    norm=json.loads((out/'C52-normalization.json').read_text());means=np.asarray(norm['mean'],np.float32);scales=np.asarray(norm['scale'],np.float32)
    heads={}
    for name in ('C52-ALIGNED','C52-SHUFFLED'):
       m=c52.Head();m.load_state_dict(load_file(str(out/(name+'.safetensors'))));m.eval();heads[name]=m
    chosen=[];seen=set()
    for row in data:
       if row['arm']!='backtrack' or row['strength']!=1 or row['graph'] in seen:continue
       o=row['outcomes'];contrast=sum(o[m]['goal']!=o['C34-RESET']['goal'] for m in c52.POLICIES)
       if contrast>0:
          chosen.append(row);seen.add(row['graph'])
       if len(chosen)>=2:break
    for row in data:
       if row['arm']=='backtrack' and row['strength']==1 and row['graph'] not in seen:
          chosen.append(row);seen.add(row['graph'])
       if len(chosen)>=3:break
    if not chosen:raise AssertionError('no eligible executed gallery rows')
    C=6;R=len(chosen); W=286;H=290;cw=16
    pieces=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{C*W}" height="{R*H+65}" viewBox="0 0 {C*W} {R*H+65}">',
      '<rect width="100%" height="100%" fill="#0b1220"/>',
      '<text x="18" y="31" font-family="sans-serif" font-size="20" fill="white">C52 | actual host-executed recovery paths</text>',
      '<text x="18" y="52" font-family="sans-serif" font-size="12" fill="#b1c3d3">Solid = executed suffix; faint = host-executed prefix; squares = protected cells; circle = new goal</text>']
    receipts=[]
    for r,row in enumerate(chosen):
      w,info=c50.c47.c42.world(row['seed'],row['family'],row['case'],row['personality'])
      got=c50.prefix(base,w,row['seed'],row['cue_tick']);assert got
      snap,rng,receipt=got;target=c50.resolved_goal(snap,info)
      assert target==tuple(row['target']) and receipt['action_sha']==row['prefix_sha']
      for c,name in enumerate(c52.POLICIES):
        z=c52.run_suffix(base,opt,prior,snap,rng,name,(row['arm'],row['strength']),row['cue_tick'],target,heads,means,scales,True)
        expected=row['outcomes'][name]
        assert (z['action_sha'],z['ink_sha'],z['goal'])==(expected['action_sha'],expected['ink_sha'],expected['goal'])
        receipts.append({'graph':row['graph'],'model':name,'action_sha':z['action_sha'],'ink_sha':z['ink_sha'],'goal':z['goal']})
        x=c*W;y=65+r*H
        pieces.append(f'<rect x="{x+1}" y="{y+1}" width="{W-2}" height="{H-2}" rx="8" fill="#142338" stroke="#263a51"/>')
        pieces.append(f'<text x="{x+12}" y="{y+22}" font-family="sans-serif" font-size="15" fill="{COLORS[name]}">{name}</text>')
        pieces.append(f'<text x="{x+12}" y="{y+41}" font-family="sans-serif" font-size="11" fill="#c6d6e7">{row["family"]} / {row["case"]} / cue t={row["cue_tick"]} / goal={z["goal"]}</text>')
        left=x+20;top=y+50
        pieces.append(f'<rect x="{left}" y="{top}" width="240" height="240" fill="#0c1726"/>')
        for p in snap.protected:
          pieces.append(f'<rect x="{left+p[0]*cw}" y="{top+p[1]*cw}" width="{cw}" height="{cw}" fill="#566170" opacity=".75"/>')
        def poly(points,stroke,sw,opacity):
          if len(points)<2:return ''
          coords=' '.join(f'{left+cw*(p[0]+.5):.1f},{top+cw*(p[1]+.5):.1f}' for p in points)
          return f'<polyline points="{coords}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        pieces.append(poly(receipt['positions'],'#bdd0e8',2,.28))
        pts=[snap.pos]+[tuple(p['pos']) for p in z['path']]
        pieces.append(poly(pts,COLORS[name],2.5,.95))
        pieces.append(f'<circle cx="{left+cw*(target[0]+.5)}" cy="{top+cw*(target[1]+.5)}" r="5.8" fill="#f9e99b"/>')
        pieces.append(f'<circle cx="{left+cw*(snap.pos[0]+.5)}" cy="{top+cw*(snap.pos[1]+.5)}" r="3.8" fill="white"/>')
    pieces.append('</svg>')
    (out/'C52-gallery.svg').write_text('\n'.join(pieces)+'\n')
    (out/'C52-gallery-replay-receipts.json').write_text(json.dumps(receipts,sort_keys=True,indent=2)+'\n')
    print('verified gallery paths',len(receipts))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('out');a=p.parse_args();build(a.root,a.out)
