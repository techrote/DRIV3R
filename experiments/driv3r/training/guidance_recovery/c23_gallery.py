"""Read-only SVG gallery from actual C23 synthetic host-executed traces."""
from __future__ import annotations
import html,json,sys
from pathlib import Path
from c22_branch_personality import world

def panel(r,seed):
    idx=(r['seed']-seed-(100000 if r['outcome']=='ahead' else 0))//5099
    w=world(r['seed'],r['family'],r['personality'],r['outcome'],
            ((idx//2)+2)%4,not bool((idx//8)%2),2,idx*13)
    square=12;offset=18
    rect=[]
    for y in range(15):
      for x in range(15):
        pos=(x,y)
        shade='#394458' if pos in w.protected else '#d6e3e4'
        if pos in w.inked:shade='#92c8a2'
        rect.append(f'<rect x="{offset+x*square}" y="{offset+y*square}" width="11" height="11" fill="{shade}"/>')
    path=[w.pos]+[tuple(t['cursor']) for t in r['trace']]
    points=' '.join(f'{offset+x*square+5},{offset+y*square+5}' for x,y in path)
    gx,gy=w.goal;px,py=w.pos
    label=f"{r['model']} · {r['outcome']} · {r['directive']} · hint {r['arm']} · cue {r['strength']} · personality {r['personality']:+}"
    return '<section><h3>'+html.escape(label)+'</h3><svg viewBox="0 0 215 215" aria-label="Source reconstructed synthetic host executed path">'+''.join(rect)+f'<polyline points="{points}" stroke="#f28e49" stroke-width="2.7" fill="none"/>' +f'<circle cx="{offset+gx*square+5}" cy="{offset+gy*square+5}" r="5" fill="#1684d0"/><circle cx="{offset+px*square+5}" cy="{offset+py*square+5}" r="4" fill="#db3848"/></svg>'+f'<p>{"GOAL" if r["reached"] else "missed"} · {r["steps"]} decisions · seed {r["seed"]}</p></section>'

def render(rows,seed):
    keys=[]
    for r in rows:
        if not r['trace']:continue
        key=(r['seed'],r['outcome'],r['personality'],r['arm'],r['directive'],r['strength'])
        if key not in keys and len(keys)<8:keys.append(key)
    chosen=[r for r in rows if r['trace'] and (r['seed'],r['outcome'],r['personality'],r['arm'],r['directive'],r['strength']) in keys]
    return '<!doctype html><html lang="en"><meta charset="utf-8"><title>C23 executed recovery paths</title><style>body{font:14px system-ui;background:#131b2a;color:#ebf2f5;margin:20px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}section{background:#283348;border-radius:8px;padding:10px}h3{font-size:12px;min-height:32px}svg{width:100%}p{font-size:12px;color:#d2dae2}</style><h1>C23 matched recovery: actual synthetic trajectories</h1><p>Five policies under matched seeds; protected cells are slate, accessible cells pale, initial ink green, executed trajectory orange, goal blue, initial cursor red. Synthetic fixture only — not a CyberSand runtime capture.</p><main>'+''.join(panel(r,seed) for r in chosen)+'</main></html>'

if __name__=='__main__':
    src=Path(sys.argv[1]);dest=Path(sys.argv[2]);seed=int(sys.argv[3]);rows=json.loads(src.read_text())
    dest.write_text(render(rows,seed),encoding='utf-8')
