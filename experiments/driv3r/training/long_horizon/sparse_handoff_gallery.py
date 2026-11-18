"""Render original action-receipt trajectories; never generate imagined paths."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape
from .sparse_handoff_lab import fixture
from .world import Spec

CASES=(('pickets','original',1,'neural'),('pickets','original',1,'stall8'),
       ('islands','original',0,'neural'),('islands','original',0,'stall8'),
       ('islands','east_sleeve',0,'neural'),('islands','east_sleeve',0,'late24'))

def draw(source,output):
    data=[]
    for family,layout,gi,arm in CASES:
        matches=[]
        for file in (source/'traces').glob('*.json'):
            if f'-{family}-' in file.name and f'-{layout}-5-{gi}-{arm}.json' in file.name and file.name.startswith('711-'):
                matches.append(file)
        assert len(matches)==1,(family,layout,gi,arm,matches)
        tr=json.loads(matches[0].read_text())
        w=fixture(Spec(**tr['spec']),tr['row']['depth'],layout)
        pts=[ev['cursor_before'] for ev in tr['prelude']]+[tr['prelude'][-1]['cursor_after']]
        pts.extend(ev['receipt']['cursor_after'] for ev in tr['entries'])
        data.append((tr,w,pts))
    out=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1290 890">',
         '<rect width="1290" height="890" fill="#f8fafc"/>',
         '<text x="28" y="34" font-family="sans-serif" font-size="23" font-weight="bold">DRIV3R · sparse planner handoff / DEVELOPMENT receipt gallery</text>',
         '<text x="28" y="55" font-family="sans-serif" font-size="12" fill="#475569">Source-matched synthetic action histories · teal = executed path · salmon = material/protected · no native CyberSand claims</text>']
    for i,(tr,w,pts) in enumerate(data):
        col,row=i%2,i//2; ox=28+col*637; oy=75+row*266
        r=tr['row']; goal=r['goal']
        allx=[p[0] for p in pts]+[goal[0]]; ally=[p[1] for p in pts]+[goal[1]]
        xmin,xmax=max(0,min(allx)-3),min(95,max(allx)+3)
        ymin,ymax=max(0,min(ally)-3),min(95,max(ally)+3)
        unit=min(573/(xmax-xmin+1),205/(ymax-ymin+1),10.2)
        out.append(f'<g transform="translate({ox},{oy})">')
        out.append('<rect x="0" y="0" width="602" height="247" rx="10" fill="white" stroke="#cbd5e1"/>')
        cap=f"{r['family']} / {r['layout']} · {r['arm']} · {r['reason']} · {r['decisions']} actions · {r['planner_calls']} searches"
        out.append('<text x="12" y="20" font-family="monospace" font-size="12">'+escape(cap)+'</text>')
        out.append('<g transform="translate(12,27)">')
        for px,py in list(w.materials)+list(w.protected):
            if xmin<=px<=xmax and ymin<=py<=ymax:
                x=(px-xmin)*unit; y=(py-ymin)*unit
                out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{unit:.1f}" height="{unit:.1f}" fill="#f1b9ad" opacity=".7"/>')
        coords=' '.join(f'{(p[0]-xmin+.5)*unit:.1f},{(p[1]-ymin+.5)*unit:.1f}' for p in pts)
        out.append(f'<polyline points="{coords}" fill="none" stroke="#0f766e" stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round"/>')
        for p,color in ((pts[0],'#0369a1'),(pts[-1],'#7c3aed'),(goal,'#b45309')):
            out.append(f'<circle cx="{(p[0]-xmin+.5)*unit:.1f}" cy="{(p[1]-ymin+.5)*unit:.1f}" r="4.2" fill="{color}" stroke="white" stroke-width="1.2"/>')
        out.append('</g></g>')
    out.append('<text x="28" y="880" font-family="sans-serif" font-size="11" fill="#64748b">Blue start · violet final cursor · amber target · complete forced prelude included · 64 logical-move synthetic gate</text></svg>')
    output.write_text('\n'.join(out)+'\n')
    return output

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    args=a.parse_args();print(draw(args.run,args.out))
