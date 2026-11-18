"""Render source-matched C64 executed movement traces (no inferred maze walls)."""
from __future__ import annotations
import json, sys, html
from pathlib import Path

COLS=5;W=220;H=168
POL=('C34-RESET','C60-FRONTIER','C60-FROZEN','C64-ALIGNED','C64-SHUFFLED')

def render(traces):
    data={}
    for t in traces:
        k=(t['family'],t['graph'])
        if k not in data:data[k]={}
        data[k][t['policy']]=t
    keys=[];families=[]
    for key in data:
        if key[0] not in families:
            families.append(key[0]);keys.append(key)
    width=W*COLS;height=H*len(keys)
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
       '<rect width="100%" height="100%" fill="#f6f6f6"/>',
       '<style>text{font:11px monospace;fill:#161616}.small{font:9px monospace;fill:#444}</style>']
    for row,key in enumerate(keys):
        allpts=[]
        for p in POL:
            t=data[key][p]
            allpts.extend([t['start'],t['target']]+[q['p'] for q in t['path']])
        xmin=min(p[0] for p in allpts)-1;xmax=max(p[0] for p in allpts)+1
        ymin=min(p[1] for p in allpts)-1;ymax=max(p[1] for p in allpts)+1
        scale=min(187/max(1,xmax-xmin),118/max(1,ymax-ymin))
        def coord(p,x,y):return (x+14+(p[0]-xmin)*scale,y+28+(p[1]-ymin)*scale)
        for ci,p in enumerate(POL):
            t=data[key][p];x=ci*W;y=row*H
            s.append(f'<rect x="{x+2}" y="{y+2}" width="{W-4}" height="{H-4}" rx="4" fill="white" stroke="#b8b8b8"/>')
            label=f'{key[0].upper()} · {p}'
            s.append(f'<text x="{x+8}" y="{y+17}">{html.escape(label)}</text>')
            # Render only the path really executed; no fabricated obstacles.
            ps=[t['start']]+[q['p'] for q in t['path']]
            line=' '.join(f'{coord(z,x,y)[0]:.1f},{coord(z,x,y)[1]:.1f}' for z in ps)
            col='#2866a1' if t['goal'] else '#bb5638'
            s.append(f'<polyline points="{line}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
            tx,ty=coord(t['target'],x,y);sx,sy=coord(t['start'],x,y)
            s.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="3.5" fill="#2b2b2b"/>')
            s.append(f'<path d="M{tx-4:.1f},{ty:.1f}h8m-4,-4v8" stroke="#288247" stroke-width="1.8"/>')
            s.append(f'<text class="small" x="{x+8}" y="{y+H-11}">goal {t["goal"]} · {len(t["path"])} actions · {t["action_sha"][:10]}</text>')
    s.append('</svg>')
    return '\n'.join(s)+'\n'

if __name__=='__main__':
    src,dst=map(Path,sys.argv[1:3])
    Path(dst).write_text(render(json.loads(src.read_text())))
