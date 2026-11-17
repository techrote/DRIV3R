"""Static, source-derived C29 executed-trajectory gallery; no fabricated paths."""
from __future__ import annotations
import argparse,html,json
from pathlib import Path
from c28_temporal_option import executed_world
from c29_deepdecoy_test import deep_world

MODELS=('C18-A0','VETO','PROBE','GUARD')
CASES=(('reverse','backtrack'),('tunnel','continue'),('decoy','backtrack'),('deepdecoy','backtrack'))

def panel(r):
    step=14
    world=(deep_world(r['i'],r['personality'],r['seed']) if r['outcome']=='deepdecoy'
        else executed_world(r['i'],r['outcome'],r['personality'],r['seed'],False))
    assert r['prefix_sha256']
    rec=[]
    for x,y in sorted(world.protected):rec.append(f'<rect x="{x*step}" y="{y*step}" width="{step}" height="{step}" fill="#454b54"/>')
    for x,y in sorted(world.inked):rec.append(f'<rect x="{x*step+5}" y="{y*step+5}" width="4" height="4" fill="#8ca0a5"/>')
    pts=[world.pos]+[tuple(p['pos']) for p in r['path']]
    points=' '.join(f'{(x+.5)*step},{(y+.5)*step}' for x,y in pts)
    rec.append(f'<polyline points="{points}" stroke="#ebbb73" stroke-width="2.2" fill="none" stroke-linejoin="round" stroke-linecap="round"/>')
    sx,sy=world.pos;gx,gy=world.goal
    rec.append(f'<circle cx="{(sx+.5)*step}" cy="{(sy+.5)*step}" r="5" fill="#56baff"/>')
    rec.append(f'<circle cx="{(gx+.5)*step}" cy="{(gy+.5)*step}" r="5" fill="#68e0a0"/>')
    return (f'<article><b>{html.escape(r["model"])} · {"PASS" if r["success"] else "FAIL"}</b>'
        f'<svg viewBox="0 0 210 210" role="img" aria-label="Executed path"><rect width="210" height="210" fill="#141920"/>{"".join(rec)}</svg>'
        f'<p>{html.escape(r["outcome"])} / {html.escape(r["directive"])} · {r["steps"]} decisions · {r["backtracks"]} backtracks</p></article>')

def build(src,dest):
    src=Path(src);rows=json.loads((src/'cohort_a.json').read_text())+json.loads((src/'deepdecoy_a.json').read_text());chosen=[]
    for outcome,directive in CASES:
        for name in MODELS:
            matches=[r for r in rows if r['outcome']==outcome and r['directive']==directive and r['model']==name
                and r['i']==0 and r['personality']==1 and r['arm']==1. and r['strength']==1. and r['path']]
            assert len(matches)==1,(outcome,directive,name)
            chosen.append(matches[0])
    css='body{font-family:system-ui;background:#11151a;color:#f0f5f7;margin:24px}main{max-width:1250px;display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:12px}article{background:#232b32;border-radius:8px;padding:10px}svg{display:block;width:100%;margin-top:9px}p{font-size:12px;color:#ccd5dc}@media(max-width:800px){main{grid-template-columns:repeat(2,minmax(150px,1fr))}}'
    htmlbody=f'<!doctype html><html lang="en"><meta charset="utf-8"><title>C29 executed recovery</title><style>{css}</style><h2>DRIV3R C29 · verified executed trajectories</h2><p>16 matched panels. Protected material grey, executed path amber, start blue, goal green. External synthetic host only; not CyberSand native gameplay.</p><main>{"".join(panel(r) for r in chosen)}</main></html>'
    Path(dest).write_text(htmlbody,encoding='utf-8')
    return len(chosen)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('output');args=p.parse_args();print(build(args.results,args.output))
