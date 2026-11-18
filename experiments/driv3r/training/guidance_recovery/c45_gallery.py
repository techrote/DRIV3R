"""Render C45 host-executed synthetic trajectory evidence, not invented oracle paths."""
import argparse,html,json
from pathlib import Path

def build(traces, out):
    cards=[]
    for row in traces:
        pts=[row['initial']]+[t['position'] for t in row['trajectory']]
        xy=' '.join(f'{15+15*x},{15+15*y}' for x,y in pts)
        protected=''.join(f'<rect x="{15+15*x-5}" y="{15+15*y-5}" width="10" height="10" fill="#83919d"/>' for x,y in row['protected'])
        goal=row['trajectory'][-1]['goal']
        cards.append('<article><b>'+html.escape(f"{row['family']} / {row['model']} / {row['goals']} goals")+
          '</b><svg viewBox="0 0 260 260" xmlns="http://www.w3.org/2000/svg">'+
          '<rect width="260" height="260" fill="#11212f"/>'+protected+
          f'<polyline points="{xy}" fill="none" stroke="#fdaf53" stroke-width="2" />'+
          f'<circle cx="{15+15*pts[0][0]}" cy="{15+15*pts[0][1]}" r="4" fill="#51d7ae"/>'+
          f'<circle cx="{15+15*goal[0]}" cy="{15+15*goal[1]}" r="4" fill="#9caefb"/>'+
          '</svg><small>Actions '+str(len(row['trajectory']))+' | hash '+html.escape(row['actions_sha'][:16])+'</small></article>')
    source='''<!doctype html><html lang="en"><meta charset="utf-8"><title>C45 selective guidance executed paths</title>
<style>body{background:#0a151f;color:#e6edf4;font:14px system-ui;padding:24px}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px}
article{background:#263644;border-radius:8px;padding:12px}svg{width:100%;margin:10px 0;display:block}
small{display:block;color:#bacbd8}</style><h1>C45 — selective guidance / recovery trajectories</h1>
<p>Executable synthetic-host paths, not native CyberSand runs. Each path was executed again
with matching action and ink hashes; gray cells protected, green start, blue final goal.</p><main>'''+''.join(cards)+'</main></html>'
    Path(out).write_text(source)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('traces');p.add_argument('output');a=p.parse_args()
    build(json.loads(Path(a.traces).read_text()),a.output)
