"""Generate a static inspectable gallery from actually executed C28 paths."""
from __future__ import annotations
import html,json,sys
from pathlib import Path
from c28_temporal_option import executed_world

SELECT=(('reverse','backtrack','1.0'),('reverse','continue','1.0'),
        ('tunnel','backtrack','1.0'),('decoy','continue','1.0'),
        ('short','backtrack','1.0'))
MODELS=('C18-A0','TTL','PERSIST','VETO')

def panel(r):
    size=14
    world=executed_world(r['index'],r['outcome'],r['personality'],r['seed'],False)
    rectangles=[]
    for x,y in sorted(world.protected):
        rectangles.append(f'<rect x="{x*size}" y="{y*size}" width="{size}" height="{size}" fill="#454b54"/>')
    for x,y in sorted(world.inked):
        rectangles.append(f'<rect x="{x*size+4}" y="{y*size+4}" width="6" height="6" fill="#78929a" opacity=".46"/>')
    path=[world.pos]+[tuple(item['pos']) for item in r['path']]
    points=' '.join(f'{(x+.5)*size},{(y+.5)*size}' for x,y in path)
    rectangles.append(f'<polyline points="{points}" fill="none" stroke="#e7b767" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>')
    sx,sy=world.pos;gx,gy=world.goal
    rectangles.append(f'<circle cx="{(sx+.5)*size}" cy="{(sy+.5)*size}" r="5" fill="#55bbff"/>')
    rectangles.append(f'<circle cx="{(gx+.5)*size}" cy="{(gy+.5)*size}" r="5" fill="#69d99e"/>')
    return f'''<article><h3>{html.escape(r['model'])} · {'SUCCESS' if r['success'] else 'FAIL'}</h3>
      <svg viewBox="0 0 210 210" role="img" aria-label="Executed worm path with protected cells, start and goal">
      <rect width="210" height="210" fill="#13171c"/>{''.join(rectangles)}</svg>
      <div>{html.escape(r['outcome'])} / {html.escape(r['directive'])} · {r['steps']} steps · {r['backtracks']} retraces</div>
      <small>Real host-applied synthetic path; blue start, green goal, amber trajectory, gray protected</small></article>'''

def build(src,dest):
    rows=json.loads(Path(src).read_text())
    picked=[]
    for outcome,directive,strength in SELECT:
        for model in MODELS:
            match=[r for r in rows if r['index']==0 and r['personality']==1 and r['arm']==1.
                and r['outcome']==outcome and r['directive']==directive and r['strength']==float(strength)
                and r['model']==model and r['path']]
            if not match:raise ValueError((outcome,directive,model))
            picked.append(match[0])
    css='''body{background:#101217;color:#edf1f7;font-family:system-ui;margin:24px;max-width:1180px}h1{margin-bottom:6px}p{color:#bdc8d5}main{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:12px}article{background:#202630;padding:12px;border-radius:8px}article h3{margin:3px 0 8px;font-size:15px}svg{width:100%;border:1px solid #343a44;border-radius:4px}article div{font-size:12px;margin:8px 0}article small{color:#abb9c7;font-size:11px}@media(max-width:800px){main{grid-template-columns:repeat(2,minmax(160px,1fr))}}'''
    out=f'<!DOCTYPE html><html lang="en"><meta charset="utf-8"><title>C28 executed recovery paths</title><style>{css}</style><h1>DRIV3R C28 · executed long-horizon recovery</h1><p>20 matched held-out paths. Each column is one policy; each row shares a world and guidance condition. Research-only synthetic host, not native gameplay.</p><main>{"".join(map(panel,picked))}</main></html>'
    Path(dest).write_text(out)
    return len(picked)

if __name__=='__main__':
    assert len(sys.argv)==3
    print(build(sys.argv[1],sys.argv[2]))
