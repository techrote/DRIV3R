"""Draw selected, directly replayable synthetic trajectories; no native screenshot claim."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from xml.sax.saxutils import escape
from .world import Spec, World


def build(run,output):
    data=json.loads((Path(run)/'traces.json').read_text());panels=[]
    # Same unseen-to-training DEVELOPMENT geometry and model across columns.
    for h in (36,72,128):
        for forced in (False,True):
            rows=[x for x in data if x['summary']['horizon']==h and
                  x['summary']['arm']=='neural' and x['summary']['seed']==1051 and
                  x['summary']['family']=='pickets' and x['summary']['challenge']=='normal' and
                  x['spec']['seed']==3000 and x['summary']['forced']==forced]
            assert len(rows)==1
            panels.append(rows[0])
    w=930;h=1160;out=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<rect width="930" height="1160" fill="#101722"/>',
     '<text x="26" y="32" fill="#f1f5f9" font-size="19" font-family="sans-serif">DRIV3R — original synthetic 36 / 72 / 128 legal-action traces</text>',
     '<text x="26" y="53" fill="#b4c3d1" font-size="12" font-family="sans-serif">Solid cyan: logical moves ≤ 64  ·  Dashed orange: beyond 64  ·  No native game execution</text>']
    for i,case in enumerate(panels):
        row,col=divmod(i,2);x0=26+col*455;y0=79+row*357;scale=3.24
        s=Spec(**case['spec']);base=World.create(s);ev=case['events'];stats=case['summary']
        out.append(f'<g transform="translate({x0},{y0})">')
        out.append('<rect x="0" y="0" width="433" height="342" rx="6" fill="#1c2736" stroke="#43546c"/>')
        label=f"{stats['horizon']} decisions · {'perturbed' if stats['forced'] else 'ordinary'} · model 1051"
        out.append(f'<text x="13" y="21" fill="#fff" font-family="sans-serif" font-size="14">{escape(label)}</text>')
        out.append(f'<text x="13" y="38" fill="#b4c3d1" font-family="sans-serif" font-size="11">executed {stats["steps"]} · paint {stats["painted"]} · cap-valid {stats["prefix_64_decisions"]} decisions · forced Δ {stats["forced_changed"]}</text>')
        out.append('<g transform="translate(48,52)">')
        out.append(f'<rect x="0" y="0" width="{96*scale}" height="{96*scale}" fill="#101927"/>')
        for typ,color in ((37,'#55799e'),(1,'#475463')):
            rects=[f'M{x*scale:.2f},{y*scale:.2f}h{scale:.2f}v{scale:.2f}h{-scale:.2f}z'
                   for (x,y),kind in base.materials.items() if kind==typ]
            if rects:out.append(f'<path d="{" ".join(rects)}" fill="{color}"/>')
        rects=[f'M{x*scale:.2f},{y*scale:.2f}h{scale:.2f}v{scale:.2f}h{-scale:.2f}z'
               for x,y in sorted(base.protected)]
        if rects:out.append(f'<path d="{" ".join(rects)}" fill="#af72ab"/>')
        prev=(48,48);good=[];excess=[];first_beyond=None
        for e in ev:
            here=tuple(e['cursor_after']);segment=(prev,here)
            if e['logical_moves']<=64:good.append(segment)
            else:
                excess.append(segment)
                if first_beyond is None:first_beyond=prev
            prev=here
        for segments,color,dash in [(good,'#34d5c5',''),(excess,'#ffad52',' stroke-dasharray="3 2"')]:
            for (ax,ay),(bx,by) in segments:
                if (ax,ay)==(bx,by):continue
                out.append(f'<line x1="{(ax+.5)*scale:.2f}" y1="{(ay+.5)*scale:.2f}" x2="{(bx+.5)*scale:.2f}" y2="{(by+.5)*scale:.2f}" stroke="{color}" stroke-width="2"{dash}/>' )
        out.append(f'<circle cx="{48.5*scale:.2f}" cy="{48.5*scale:.2f}" r="3.7" fill="#ffffff"/>')
        if first_beyond:
            x,y=first_beyond
            out.append(f'<circle cx="{(x+.5)*scale:.2f}" cy="{(y+.5)*scale:.2f}" r="5.0" fill="none" stroke="#ffad52" stroke-width="2"/>')
        if ev:
            x,y=ev[-1]['cursor_after']
            out.append(f'<circle cx="{(x+.5)*scale:.2f}" cy="{(y+.5)*scale:.2f}" r="4.4" fill="#fb7185"/>')
        out.append('</g></g>')
    out.append('</svg>');Path(output).write_text('\n'.join(out)+'\n')
    return str(output)
if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--run',required=True);pa.add_argument('--out',required=True)
    print(build(pa.parse_args().run,pa.parse_args().out))
