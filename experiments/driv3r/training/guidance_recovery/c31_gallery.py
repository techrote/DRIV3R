"""C31 executed trajectory gallery (static, source-derived SVG)."""
from __future__ import annotations
import argparse,hashlib,json,html
from pathlib import Path
from c31_observed_progress import u_world

def make(cohort_file,out,seed):
    rows=json.loads(Path(cohort_file).read_text())
    relevant=[r for r in rows if r['path'] and r['directive']=='backtrack' and
              r['strength']==1. and r['model'] in ('C18-A0','C29-PROBE','C31-RULE','C31-LEARNED')]
    group=[]
    for outcome in ('decoy2','decoy3','decoy4','tunnel4'):
        selected=[r for r in relevant if r['outcome']==outcome and r['personality']==1 and r['arm']==1.]
        group.extend(selected)
    cells=[]
    for r in group:
        kind='tunnel' if r['outcome']=='tunnel4' else 'decoy';depth=int(r['outcome'][-1])
        w=u_world(r['i']+8,kind,r['personality'],r['seed'],depth)
        protected=set(map(tuple,w.protected));path=[w.pos]+[tuple(p['pos']) for p in r['path']]
        goal=w.goal;parts=[];side=17;pad=16
        for x,y in protected:
            parts.append(f'<rect x="{pad+x*side}" y="{pad+y*side}" width="{side}" height="{side}" fill="#3c4350"/>')
        points=' '.join(f'{pad+side*(x+.5):.1f},{pad+side*(y+.5):.1f}' for x,y in path)
        parts.append(f'<polyline points="{points}" fill="none" stroke="#49cdf8" stroke-width="2.2" stroke-linejoin="round" opacity=".86"/>')
        gx,gy=goal;parts.append(f'<circle cx="{pad+side*(gx+.5)}" cy="{pad+side*(gy+.5)}" r="5" fill="#ffb700"/>')
        first=path[0];parts.append(f'<circle cx="{pad+side*(first[0]+.5)}" cy="{pad+side*(first[1]+.5)}" r="4" fill="#ec6a7a"/>')
        svg=f'<svg viewBox="0 0 287 287" xmlns="http://www.w3.org/2000/svg" role="img"><rect width="287" height="287" fill="#151a22"/>{"".join(parts)}</svg>'
        cells.append('<article><h3>'+html.escape(r['outcome']+' / '+r['model'])+'</h3>'+svg+
                     f'<p>Goal {"reached" if r["success"] else "missed"} · {r["steps"]} actions · host rejections {r["host_rejected"]}</p></article>')
    document='<!doctype html><meta charset="utf-8"><title>C31 executed policy traces</title><style>body{background:#111821;color:#e6edf5;font:14px system-ui;margin:30px}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:15px}article{background:#1b2635;padding:12px;border-radius:9px}svg{width:100%;height:auto}p{color:#bac7d9}</style><h1>C31 host-executed trails</h1><p>U-corridor withheld geometry: slate cells are protected, cyan lines are recorded actions, amber is the changed goal, coral is the starting point. No plans or oracle paths are rendered.</p><section>'+''.join(cells)+'</section>'
    Path(out).write_text(document)
    return {'panels':len(cells),'html_sha256':hashlib.sha256(document.encode()).hexdigest()}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('rows');p.add_argument('output');p.add_argument('--seed',type=int,default=311111)
    a=p.parse_args();print(make(a.rows,a.output,a.seed))
