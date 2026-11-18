"""C37 executed trajectory gallery with legal-host replay and hash verification."""
from __future__ import annotations
import argparse, hashlib, html, json
from pathlib import Path
from experiment import sha
from c34_independent_maze import make_world
from c36_gallery import verify

PRESENT=('C34-RESET','C37-LEARNED','C37-SHUFFLED','C37-RANDOM')


def render(rows,output):
    output=Path(output)
    selected=[];families=set()
    for row in rows:
        if row['trace'] is None or row['directive']!='backtrack' or row['strength']!=1.:
            continue
        group=(row['family'],row['case'],row['switch'],row['seed'])
        if group not in families:
            if len(families)>=4:continue
            families.add(group)
        if row['model'] in PRESENT:selected.append(row)
    assert len(selected)==16,(len(selected),families)
    parts=['<!DOCTYPE html><html><head><meta charset="utf-8"><title>C37 host-executed gallery</title>',
      '<style>body{font:14px system-ui;background:#131a22;color:#e5ebf4;margin:24px}h1{font-size:23px}.grid{display:grid;grid-template-columns:repeat(4,minmax(190px,1fr));gap:14px}figure{background:#203040;border:1px solid #415162;padding:10px;margin:0}svg{width:100%;height:auto}figcaption{font-size:12px;line-height:1.4}</style>',
      '</head><body><h1>C37: matched executed trajectories</h1><p>White: legal terrain. Dark: protected cells. Teal: host-executed prefix. Orange: actual sampled policy decisions. Magenta: active goal. Research-only synthetic host; every panel independently replayed against its action and ink hashes.</p><div class="grid">']
    for row in selected:
        w,coords,info=verify(row)
        cells={(x,y) for x in range(15) for y in range(15) if (x,y) not in w.protected}
        rectangles=''.join(f'<rect x="{x*10}" y="{y*10}" width="10" height="10" fill="#e0e5df"/>' for x,y in cells)
        wi,_=make_world(row['seed'],row['family'],row['case'],row['personality'])
        prefix=[wi.events[0][1]]+[e[2] for e in wi.events]
        pp=' '.join(f'{x*10+5},{y*10+5}' for x,y in prefix)
        path=' '.join(f'{x*10+5},{y*10+5}' for x,y in coords)
        x,y=w.goal
        parts.append(f'<figure><svg viewBox="0 0 150 150">{rectangles}'
          f'<polyline points="{pp}" fill="none" stroke="#17bed0" stroke-width="1.6"/>'
          f'<polyline points="{path}" fill="none" stroke="#ef8f40" stroke-width="1.8"/>'
          f'<circle cx="{10*x+5}" cy="{10*y+5}" r="3.6" fill="#ce4ace"/></svg>'
          f'<figcaption>{html.escape(str(row["model"]))}<br>{html.escape(str(row["family"]+" · "+row["case"]+" · "+("switch" if row["switch"] else "single")))}<br>goals={row["goals"]} · steps={row["steps"]}</figcaption></figure>')
    parts.append('</div></body></html>')
    output.write_text('\n'.join(parts)+'\n')
    return {'panels':len(selected),'verified_action_and_ink_hashes':len(selected),
         'source_rows_sha256':sha(selected),
         'html_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('rows');p.add_argument('output');a=p.parse_args()
    print(json.dumps(render(json.loads(Path(a.rows).read_text()),a.output),indent=2))
