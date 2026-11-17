"""Gallery of actual executed C38 paths; never substitutes oracle routes."""
from pathlib import Path
import html,json
from c34_independent_maze import make_world

ROOT=Path(__file__).resolve().parent.parent
EVIDENCE=(ROOT/'evidence') if (ROOT/'evidence'/'C38-cohort-A.json').exists() else (ROOT/'output')
MODEL=('C34-RESET','C37-LEARNED','C38-SIGN','C38-PERMUTED')

def generate(out):
    cards=[]
    for cohort in 'ABC':
        rows=json.loads((EVIDENCE/f'C38-cohort-{cohort}.json').read_text())
        q=next(r for r in rows if r['family']=='tree' and r['case']=='detour' and r['personality']==1 and r['directive']=='backtrack' and r['strength']==1.)
        for name in MODEL:
            r=next(z for z in rows if z['model']==name and z['seed']==q['seed'] and z['directive']=='backtrack')
            w,info=make_world(q['seed'],'tree','detour',1)
            a=r['trace'] or []
            open_cells=[(x,y) for x in range(15) for y in range(15) if (x,y) not in w.protected]
            geom=''.join(f'<rect x="{x*20}" y="{y*20}" width="20" height="20" fill="#e5ebf0"/>' for x,y in open_cells)
            first=w.pos
            points=[first]+[tuple(x['pos']) for x in a]
            path=' '.join(f'{x*20+10},{y*20+10}' for x,y in points)
            gx,gy=w.goal
            cards.append(f'''<section><h3>{html.escape(cohort)} · {html.escape(name)}</h3><p>Goal events: {r['goals']}, steps: {r['steps']} · action hash {r['action_sha'][:12]}</p><svg viewBox="0 0 300 300" aria-label="Host-executed path"><rect width="300" height="300" fill="#566274"/>{geom}<polyline points="{path}" fill="none" stroke="#bd3c2c" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/><circle cx="{first[0]*20+10}" cy="{first[1]*20+10}" r="6" fill="#127ab8"/><circle cx="{gx*20+10}" cy="{gy*20+10}" r="7" fill="#16a45b"/></svg></section>''')
    out=Path(out)
    out.write_text('''<!doctype html><html><head><meta charset="utf-8"><title>C38 executed trajectories</title><style>body{font:14px system-ui;background:#131820;color:#e5e7eb;margin:2rem}article{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:16px}section{background:#26303c;padding:14px;border-radius:10px}svg{width:100%;max-width:350px}h3{margin:4px}p{color:#b0bfca}</style></head><body><h1>C38 matched legal trajectories</h1><p>Grey: blocked · pale: legal · red: host-executed path · blue: start · green: first goal. Cohorts A/B/C; same scenario and seed within each row. Trace ends at the research horizon; this is not an oracle route.</p><article>'''+''.join(cards)+'''</article></body></html>''')
    return len(cards)

if __name__=='__main__':
    print(generate(EVIDENCE/'C38-gallery.html'))
