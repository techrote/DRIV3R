"""Render actual C47 host-executed movement traces as a self-contained HTML gallery."""
from __future__ import annotations
import html,json,sys
from pathlib import Path

def draw(path,protected,initial):
    cells=set(map(tuple,protected));points=[tuple(initial)]+[tuple(q['position']) for q in path]
    extent=15;W=300;step=W/extent
    out=[f'<svg viewBox="0 0 {W} {W}" role="img" aria-label="Actual executed worm positions">',
         f'<rect width="{W}" height="{W}" fill="#10171c"/>']
    for x,y in cells:
        out.append(f'<rect x="{x*step:.2f}" y="{y*step:.2f}" width="{step:.2f}" height="{step:.2f}" fill="#28393b"/>')
    if points:
        pp=' '.join(f'{(p[0]+.5)*step:.2f},{(p[1]+.5)*step:.2f}' for p in points)
        out.append(f'<polyline points="{pp}" fill="none" stroke="#f0bd4d" stroke-width="2.4" opacity=".89" stroke-linejoin="round"/>')
        x,y=points[0];out.append(f'<circle cx="{(x+.5)*step}" cy="{(y+.5)*step}" r="4" fill="#4bed96"/>')
        x,y=points[-1];out.append(f'<circle cx="{(x+.5)*step}" cy="{(y+.5)*step}" r="4" fill="#f47780"/>')
    return ''.join(out+['</svg>'])

def render(rows):
    css='''body{margin:0;padding:30px;color:#e6eef0;background:#0b1115;font:14px system-ui}h1{font-size:24px}p{color:#9cadb1;line-height:1.5}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:16px}article{background:#152029;border:1px solid #344049;border-radius:9px;padding:10px}article svg{width:100%;border-radius:4px}strong{font-size:14px}small{color:#8dabaf}'''
    cards=[]
    for i,r in enumerate(rows):
        cards.append('<article><strong>'+html.escape(r['model'])+'</strong> · goal events '+str(r['goal_events'])+
            '<div>'+draw(r['trajectory'],r['protected'],r['initial'])+'</div><small>Graph '+html.escape(r['graph'][:12])+
            ' · '+str(len(r['trajectory']))+' actions · executed SHA '+html.escape(r['actions_sha'][:12])+'</small></article>')
    return '<!doctype html><html><head><meta charset="utf-8"><title>C47 executed trajectories</title><style>'+css+'</style></head><body><h1>C47 · Executed recovery trajectories</h1><p>Research-only synthetic host; all paths are replayed, hashed, host-executed trajectories, NOT native CyberSand footage. Start is green, end red; dark cells are host protected. Selection policy evidence is retained separately in C47-online.json.</p><div class="grid">'+''.join(cards)+'</div></body></html>'

if __name__=='__main__':
    rows=json.loads(Path(sys.argv[1]).read_text());Path(sys.argv[2]).write_text(render(rows))
