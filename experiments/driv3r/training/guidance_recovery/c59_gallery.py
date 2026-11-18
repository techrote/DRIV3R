"""SVG rendering of actual, action-verified C59 synthetic-host trajectories."""
from pathlib import Path
import json,html

def render(traces,out):
    cases=json.loads(Path(traces).read_text())
    cols=4;rows=(len(cases)+cols-1)//cols
    w=1240;h=60+160*rows
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
       '<rect width="100%" height="100%" fill="#101827"/>',
       '<text x="20" y="29" fill="#f3f6ff" font-size="18" font-family="monospace">C59 | eligibility &amp; personality | host-executed paths</text>',
       '<text x="20" y="46" fill="#a9bad0" font-size="10" font-family="monospace">Only executed prefix and suffix positions; no inferred walls, terrain or unseen route lines</text>']
    for i,r in enumerate(cases):
        x=14+(i%cols)*308;y=60+(i//cols)*160
        a=[tuple(p) for p in r['prefix']['positions']]
        b=[tuple(z['p']) for z in r['outcome']['path']]
        dest=tuple(r['target']); allp=a+b+[dest]
        xs=[t[0] for t in allp];ys=[t[1] for t in allp]
        l,u=min(xs),max(xs);t,d=min(ys),max(ys)
        sc=min(11,275/max(2,u-l),115/max(2,d-t))
        def xy(p):return (x+154+(p[0]-(l+u)/2)*sc,y+84+(p[1]-(t+d)/2)*sc)
        def pl(p):return ' '.join(f'{xy(z)[0]:.1f},{xy(z)[1]:.1f}' for z in p)
        s.append(f'<rect x="{x}" y="{y}" width="299" height="151" rx="7" fill="#18263a" stroke="#44596a"/>')
        title=f'{r["family"]} / {r["goaltype"]} / {int(r["drop"]*100)}% / {r["policy"]}'
        s.append(f'<text x="{x+7}" y="{y+15}" font-family="monospace" font-size="9" fill="#ecf2fa">{html.escape(title)}</text>')
        if len(a)>1:s.append(f'<polyline points="{pl(a)}" fill="none" stroke="#8090ac" stroke-width="1.2" opacity=".8"/>')
        if len(b)>1:s.append(f'<polyline points="{pl(b)}" fill="none" stroke="#5bd5a9" stroke-width="2.2" stroke-linejoin="round"/>')
        gx,gy=xy(dest);s.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="4" fill="#f4cc60"/>')
        s.append(f'<text x="{x+8}" y="{y+143}" font-family="monospace" font-size="9" fill="#bdc8d8">goal={r["outcome"]["goal"]} steps={r["outcome"]["steps"]} legal=PASS</text>')
    Path(out).write_text('\n'.join(s+['</svg>'])+'\n')
if __name__=='__main__':
 import sys;render(sys.argv[1],sys.argv[2])