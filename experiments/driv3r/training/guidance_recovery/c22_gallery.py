"""C22 executed-trajectory SVG gallery; no unexecuted planned paths."""
from __future__ import annotations
import json,sys,html
from pathlib import Path
from c22_branch_personality import world


def draw(row,i):
    source_i=(row['seed']-340212-(100000 if row['outcome']=='ahead' else 0))//5099
    w=world(row['seed'],row['family'],row['personality'],row['outcome'],((source_i//2)+2)%4,
            not bool((source_i//8)%2),2,source_i*13)
    s=12;xoff=20;yoff=20
    tiles=[]
    for y in range(15):
        for x in range(15):
            pos=(x,y);color=('#36394e' if pos in w.protected else '#d9e6e8')
            if pos in w.inked:color='#9cd3a5'
            tiles.append(f'<rect x="{xoff+x*s}" y="{yoff+y*s}" width="{s-1}" height="{s-1}" fill="{color}"/>')
    marks=[]
    trail=[w.pos]+[tuple(t['cursor']) for t in row['trace']]
    if len(trail)>1:
        path=' '.join(f'{xoff+x*s+5},{yoff+y*s+5}' for x,y in trail)
        marks.append(f'<polyline points="{path}" fill="none" stroke="#e27635" stroke-width="3" opacity=".9"/>')
    gx,gy=w.goal;marks.append(f'<circle cx="{xoff+gx*s+5}" cy="{yoff+gy*s+5}" r="5" fill="#2185bb"/>')
    ax,ay=w.pos;marks.append(f'<circle cx="{xoff+ax*s+5}" cy="{yoff+ay*s+5}" r="4" fill="#e5483c"/>')
    label=f'{row["model"]} · {row["outcome"]} · {row["directive"]} {row["strength"]:.2f} · style {row["personality"]:+}'
    outcome='GOAL' if row['reached'] else f'distance {row["distance"]}'
    return f'<section><h3>{html.escape(label)}</h3><svg viewBox="0 0 220 220" role="img" aria-label="Source reconstructed protected grid and executed cursor path">{"".join(tiles+marks)}</svg><p>{outcome} · {row["steps"]} steps · executed seed {row["seed"]}</p></section>'

def render(rows):
    # Matched panels: choose one source per directive; all four policies shown.
    chosen=[];seen=set()
    for r in rows:
        if not r['trace']:continue
        k=(r['outcome'],r['directive'],r['strength'],r['seed'])
        if k not in seen and len(seen)<9:seen.add(k)
        if k in seen:chosen.append(r)
    # Up to 64 available source-matched panels; rows must contain executed trace.
    doc='<!doctype html><html lang="en"><meta charset="utf-8"><title>C22 real host-executed synthetic trajectories</title><style>body{font:14px system-ui;background:#101623;color:#e4edf2;padding:18px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}section{background:#202b3c;padding:8px;border-radius:10px}h3{font-size:13px}svg{width:100%;background:#b5c7cc}p{font-size:12px;color:#b9c9d8}</style><h1>C22 branch-rich guidance recovery</h1><p>Panels reconstruct the exact synthetic protected materials, executed trail, goal and host-applied actions; yellow/orange line is executed cursor path, blue target, red start. Synthetic, not CyberSand runtime.</p><main>'
    return doc+''.join(draw(r,i) for i,r in enumerate(chosen))+'</main></html>'

if __name__=='__main__':
    rows=json.loads(Path(sys.argv[1]).read_text());Path(sys.argv[2]).write_text(render(rows),encoding='utf-8')
