"""C56 source-matched SVG, drawn only from full re-executed host trace records."""
from __future__ import annotations
import hashlib, html, json
from pathlib import Path
from xml.sax.saxutils import escape

POLICY_ORDER=('C34-RESET','C55-ROUTE','C55-ALIGNED','C56-AUGMENTED')
COLORS={'C34-RESET':'#99a6ad','C55-ROUTE':'#0fc4ae','C55-ALIGNED':'#efb951','C56-AUGMENTED':'#e788b7'}

def make(trace_file,out):
    records=json.loads(Path(trace_file).read_text())
    assert len(records)==24 and all(records[i]['policy']==POLICY_ORDER[i%4] for i in range(len(records)))
    W,H=1440,1800;pad=24;stepx=350;stepy=284
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
       '<rect width="100%" height="100%" fill="#101920"/>',
       '<text x="28" y="35" fill="#f0f6f6" font-family="monospace" font-size="23" font-weight="bold">C56 / Source-executed memory-loss recovery</text>',
       '<text x="28" y="60" fill="#94afba" font-family="monospace" font-size="13">6 matched synthetic-host interventions × 4 policies — not a CyberSand native render</text>']
    for ix,rec in enumerate(records):
        col=ix%4;row=ix//4;ox=pad+col*stepx;oy=86+row*stepy
        outcome=rec['outcome'];pol=rec['policy']
        pth=[list(rec['initial'])]+[x['p'] for x in outcome['path']]
        prefix=[list(p) for p in rec['positions']]
        pts=pth+prefix+[list(rec['target'])]+[list(p) for p in rec['protected']]
        xs=[p[0] for p in pts];ys=[p[1] for p in pts]
        loX,hiX=min(xs)-1,max(xs)+1;loY,hiY=min(ys)-1,max(ys)+1
        sz=240/max(hiX-loX+1,hiY-loY+1)
        def xy(p):return (ox+30+(p[0]-loX)*sz,oy+30+(p[1]-loY)*sz)
        s.append(f'<rect x="{ox}" y="{oy}" width="333" height="264" rx="8" fill="#1a2830" stroke="#304552"/>')
        s.append(f'<text x="{ox+12}" y="{oy+21}" fill="{COLORS[pol]}" font-family="monospace" font-size="14" font-weight="bold">{pol}</text>')
        s.append(f'<text x="{ox+12}" y="{oy+247}" fill="#a9c0c5" font-family="monospace" font-size="11">{rec["family"]}  lost {int(rec["drop"]*100)}%  goal {outcome["goal"]}  turns {outcome["steps"]}</text>')
        for p in rec['protected']:
            x,y=xy(p)
            s.append(f'<rect x="{x-2:.2f}" y="{y-2:.2f}" width="4" height="4" fill="#48616b" opacity="0.55"/>')
        if len(prefix)>1:
            poly=' '.join(f'{x:.2f},{y:.2f}' for x,y in [xy(p) for p in prefix])
            s.append(f'<polyline points="{poly}" fill="none" stroke="#758a91" stroke-width="2" opacity="0.65"/>')
        if len(pth)>1:
            poly=' '.join(f'{x:.2f},{y:.2f}' for x,y in [xy(p) for p in pth])
            s.append(f'<polyline points="{poly}" fill="none" stroke="{COLORS[pol]}" stroke-width="2.8" stroke-linejoin="round" stroke-linecap="round"/>')
        x,y=xy(rec['target']);s.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#f4e7af"/>')
        x,y=xy(rec['positions'][-1]);s.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.8" fill="#ffffff"/>')
    s.append('</svg>')
    p=Path(out);p.write_text('\n'.join(s)+'\n')
    return hashlib.sha256(p.read_bytes()).hexdigest()

if __name__=='__main__':
    import sys
    print(make(sys.argv[1],sys.argv[2]))
