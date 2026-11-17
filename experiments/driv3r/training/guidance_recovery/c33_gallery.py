"""C33 actual executed-trajectory SVG: reconstruct graph from exact world seed."""
from pathlib import Path
import argparse,html,json
from c33_irregular_recovery import graft_world


def panel(row):
    w,_=graft_world(row['i']+8,row['scenario'],row['family'],row['personality'],row['seed'])
    cell=18;ss=[]
    for y in range(15):
        for x in range(15):
            q=(x,y);color='#455166' if q in w.protected else '#f0f1e8'
            ss.append(f'<rect x="{x*cell}" y="{y*cell}" width="{cell}" height="{cell}" fill="{color}" stroke="#b3b4b0" stroke-width=".35"/>')
    points=lambda pp:' '.join(f'{p[0]*cell+cell/2},{p[1]*cell+cell/2}' for p in pp)
    prefix=[*w.history,w.pos];path=[w.pos]+[tuple(t['pos']) for t in (row['path'] or [])]
    ss.append(f'<polyline points="{points(prefix)}" fill="none" stroke="#748195" stroke-width="2" stroke-dasharray="4 3"/>')
    ss.append(f'<polyline points="{points(path)}" fill="none" stroke="#f3a137" stroke-width="2.5"/>')
    ss.append(f'<circle cx="{w.pos[0]*cell+9}" cy="{w.pos[1]*cell+9}" r="5" fill="#4da9f7"/>')
    ss.append(f'<rect x="{w.goal[0]*cell+4}" y="{w.goal[1]*cell+4}" width="10" height="10" fill="#21c779"/>')
    caption=f"{row['family']} · {row['scenario']} · {row['model']} · {row['success']} goal / {row['steps']} actions"
    return '<article><h3>'+html.escape(caption)+'</h3><svg viewBox="0 0 270 270">'+''.join(ss)+'</svg></article>'


def build(rows,out):
    selections=[]
    for family in ('twigs','braid'):
        for scenario in ('off-early','off-mid','shortcut','lure4'):
            for model in ('C18-A0','C32-WATCH'):
                matches=[r for r in rows if r['family']==family and r['scenario']==scenario and r['model']==model
                    and r['i']==0 and r['personality']==1 and r['directive']=='backtrack' and r['strength']==1. and r['arm']==1.]
                assert len(matches)==1 and matches[0]['path'] is not None,(family,scenario,model)
                selections.append(matches[0])
    css='body{background:#17202c;color:#ecf0f7;font:14px system-ui;margin:auto;padding:24px;max-width:1200px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}article{background:#283546;padding:11px;border-radius:7px}svg{width:100%;background:white}h3{font-size:13px}p{color:#cad3df}'
    content='<!doctype html><html><head><meta charset="utf-8"><title>C33 executed branches</title><style>'+css+'</style></head><body><h1>C33 · irregular multibranch synthetic recovery</h1><p>16 host-executed trajectory comparisons. Grey dotted = legally executed input history; orange = sampled actions; green = goal; blue = start. Open regions were generated separately from the C32 corridor layout. No path is a predicted plan.</p><main>'+''.join(panel(r) for r in selections)+'</main></body></html>'
    Path(out).write_text(content)
    return selections

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('rows');p.add_argument('out');a=p.parse_args()
    x=build(json.loads(Path(a.rows).read_text()),a.out)
    print(len(x))
