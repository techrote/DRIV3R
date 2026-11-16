"""C21 source-matched SVG gallery from executed synthetic-host trajectories."""
from __future__ import annotations
import html,json,sys
from pathlib import Path
from collections import defaultdict
from c19_return_anchored import make_world
from experiment import N
from adversarial_cues import source_hash


def picture(row,world,size=240):
    sc=size/N
    el=[f'<svg viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Protected cells and executed worm path">',
        f'<rect width="{size}" height="{size}" fill="#f8f3e7"/>']
    for x,y in sorted(world.protected):
        el.append(f'<rect x="{x*sc:.1f}" y="{y*sc:.1f}" width="{sc:.1f}" height="{sc:.1f}" fill="#40505b"/>')
    pts=[world.pos]+[tuple(t['cursor']) for t in row['trace']]
    if len(pts)>1:
        coords=' '.join(f'{(x+.5)*sc:.1f},{(y+.5)*sc:.1f}' for x,y in pts)
        el.append(f'<polyline points="{coords}" fill="none" stroke="#d67a21" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/>')
    gx,gy=world.goal
    el.append(f'<rect x="{(gx+.15)*sc:.1f}" y="{(gy+.15)*sc:.1f}" width="{.7*sc:.1f}" height="{.7*sc:.1f}" fill="#25845b"/>')
    px,py=pts[-1]
    el.append(f'<circle cx="{(px+.5)*sc:.1f}" cy="{(py+.5)*sc:.1f}" r="4.1" fill="#da3443"/></svg>')
    return ''.join(el)


def render(rows,out,seed=240201,limit=16):
    pairs=defaultdict(dict)
    for r in rows:
        key=(r['source_sha256'],r['seed'],r['outcome'],r['personality'],r['directive'],r['arm'])
        pairs[key][r['model']]=r
    found=[]
    for k,p in pairs.items():
        if not all(x in p for x in ('C18-A0','C21-OPPORTUNITY')):continue
        a,b=p['C18-A0'],p['C21-OPPORTUNITY']
        if a['trace']!=b['trace']:
            category='improvement' if b['reached'] and not a['reached'] else 'regression' if a['reached'] and not b['reached'] else 'divergent'
            found.append((category,k,a,b))
    order={'improvement':0,'regression':1,'divergent':2}
    found.sort(key=lambda x:(order[x[0]],x[1]))
    selected=found[:limit]
    cards=[]
    for kind,k,a,b in selected:
        i=(a['seed']-seed-{'reverse':0,'ahead':100000,'shortcut':200000,'bridge':300000}[a['outcome']])//7499
        w=make_world(i,a['outcome'],a['personality'],a['seed'],train=False)
        assert source_hash(w)==a['source_sha256']
        sides=''.join(f'<div class="side"><b>{name} — {"goal reached" if r["reached"] else "unreached"}</b><div>{r["steps"]} decisions · {r["backtracks"]} backtracks · {r["left"]} left / {r["right"]} right turns</div>{picture(r,w)}</div>' for name,r in (('C18-A0',a),('C21-OPPORTUNITY',b)))
        cards.append('<section><h3>'+html.escape(f'{kind}: {a["outcome"]} / {a["directive"]} | personality {a["personality"]} | arm {a["arm"]} | seed {a["seed"]}')+'</h3><div class="pair">'+sides+'</div></section>')
    htmlout='''<!doctype html><html lang="en"><meta charset="utf-8"><title>C21 matched synthetic worm trajectories</title><style>body{margin:24px auto;padding:0 16px;max-width:1100px;background:#111c25;color:#ebf0ec;font:14px/1.5 system-ui}h1{font-size:24px}h3{font-size:15px}section{background:#253440;margin:14px 0;padding:14px;border-radius:8px}.pair{display:flex;gap:20px;flex-wrap:wrap}.side{flex:1 1 240px;max-width:350px}.side svg{width:100%;height:auto;border:1px solid #70818b}p{max-width:900px;color:#c4d3d8}</style><h1>C21 — independent synthetic host trajectories</h1><p>Side-by-side source-matched C18-A0 and C21-OPPORTUNITY worm executions. Dark = protected solid; orange = executed cursor; green = goal; red = final cursor. Same starting map, legal menu and RNG seed; after actions diverge, visited states differ. These are NOT CyberSand native, Workbench or GPU recordings. Gallery intentionally includes unchanged-outcome but divergent trajectories.</p>'''+''.join(cards)+'</html>'
    Path(out).write_text(htmlout,encoding='utf-8')
    return {'panels':len(selected)*2,'pair_categories':{k:sum(x[0]==k for x in selected) for k in order}}

if __name__=='__main__':
    rows=json.loads(Path(sys.argv[1]).read_text())
    print(render(rows,sys.argv[2]))
