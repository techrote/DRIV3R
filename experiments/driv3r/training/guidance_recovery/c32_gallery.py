"""Render source-derived SVG comparisons from C32 recorded host-executed paths."""
import argparse,html,json
from pathlib import Path
from c32_offstack_recovery import world

def panel(row):
    w=world(row['i']+8,row['scenario'],row['personality'],row['seed'])
    cell=16;lines=[]
    for y in range(15):
        for x in range(15):
            q=(x,y);col=('#465267' if q in w.protected else '#edf1e5')
            lines.append(f'<rect x="{x*cell}" y="{y*cell}" width="{cell}" height="{cell}" fill="{col}" stroke="#c8c8c2" stroke-width=".3"/>')
    initial=[*w.history,w.pos]
    points=lambda seq:' '.join(f'{p[0]*cell+cell/2},{p[1]*cell+cell/2}' for p in seq)
    lines.append(f'<polyline points="{points(initial)}" stroke="#7c8795" stroke-width="1.6" stroke-dasharray="3,3" fill="none"/>')
    path=[w.pos]+[tuple(t['pos']) for t in (row.get('path') or [])]
    lines.append(f'<polyline points="{points(path)}" stroke="#ef9d23" stroke-width="2.1" fill="none"/>')
    gx,gy=w.goal;lines.append(f'<rect x="{gx*cell+3}" y="{gy*cell+3}" width="10" height="10" fill="#1cba6c"/>')
    x,y=w.pos;lines.append(f'<circle cx="{x*cell+8}" cy="{y*cell+8}" r="4" fill="#2686e0"/>')
    title=f"{row['model']} · {row['scenario']} · {'PASS' if row['success'] else 'FAIL'} · {row['steps']} steps"
    return '<article><h3>'+html.escape(title)+'</h3><svg viewBox="0 0 240 240" role="img" aria-label="Executed path and blocked material">'+''.join(lines)+'</svg><p>Gray dashed: executed initial trail. Orange: policy execution. Blue: start. Green: goal.</p></article>'

def build(rows,path):
    models=('C18-A0','C32-SAFE','C32-WATCH')
    scenarios=('off-early','off-late','lure3','shortcut')
    selected=[]
    for s in scenarios:
        for m in models:
            match=[r for r in rows if r['scenario']==s and r['model']==m and r['i']==0 and r['personality']==1 and r['directive']=='backtrack' and r['strength']==1. and r['arm']==1.]
            assert len(match)==1,(s,m,len(match))
            assert match[0].get('path') is not None
            selected.append(match[0])
    css='body{background:#151c26;color:#f4f4f4;font:14px system-ui;padding:20px;margin:auto;max-width:1100px} main{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}article{background:#263141;padding:12px;border-radius:9px}svg{width:100%;background:#ddd}h3{font-size:14px}p{font-size:11px;color:#ced5e1}'
    content='<!doctype html><html><head><meta charset="utf-8"><title>C32 executed trajectory gallery</title><style>'+css+'</style></head><body><h1>C32 · Off-stack recovery, matched executed trajectories</h1><p>Synthetic host only. Every candidate action rechecked through host legality. These are actual sampled execution paths, not an oracle or planned solution.</p><main>'+''.join(panel(r) for r in selected)+'</main></body></html>'
    Path(path).write_text(content)
    return len(selected)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cohort_json');p.add_argument('output_html');a=p.parse_args()
    print(build(json.loads(Path(a.cohort_json).read_text()),a.output_html))
