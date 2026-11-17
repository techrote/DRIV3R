"""C27 offline SVG evidence from host-executed movement traces, not generated routes."""
from pathlib import Path
import json, html

D=15

def panel(r):
    points=[tuple(z['pos']) for z in r['path']]
    if r.get('prefix'):points=[tuple(r['prefix'][-1][1])]+points
    p=' '.join(f'{25+15*x},{25+15*y}' for x,y in points)
    protected=r['geometry']['protected']
    rock=''.join(f'<rect x="{18+15*x}" y="{18+15*y}" width="14" height="14" fill="#394352"/>' for x,y in protected)
    gx,gy=r['geometry']['goal'];sx,sy=points[0]
    color='#38bdf8' if r['model']=='C18-A0' else '#eab308' if 'STOP-3' in r['model'] else '#fb7185'
    return (f'<section><b>{html.escape(r["cohort"])} · {html.escape(r["model"])} · {html.escape(r["directive"])} '
      f'({r["strength"]:.2f})</b><small>Goal {r["success"]} · steps {r["steps_after_cue"]} · '
      f'end d {r["end_distance"]} · BACKTRACK {r["backtracks"]}</small>'
      f'<svg viewBox="0 0 270 270" xmlns="http://www.w3.org/2000/svg" role="img" '
      f'aria-label="Executed protected-cell grid and cursor trajectory">'
      f'<rect width="270" height="270" fill="#111827"/>{rock}'
      f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="2.4" '
      f'stroke-linecap="round" stroke-linejoin="round" opacity=".9"/>'
      f'<circle cx="{25+15*gx}" cy="{25+15*gy}" r="6" fill="#4ade80"/>'
      f'<circle cx="{25+15*sx}" cy="{25+15*sy}" r="4" fill="white"/></svg></section>')

def generate(path,out):
    rows=json.loads(Path(path).read_text());sel=[]
    for cohort in ('A','B'):
      for directive in ('backtrack','continue'):
        for model in ('C18-A0','STOP-3-DIAGNOSTIC','STOP-6-DIAGNOSTIC'):
          r=next(x for x in rows if x['cohort']==cohort and x['directive']==directive and x['model']==model and x['index']==0 and x['personality']==1 and x['arm']==1. and x['strength']==1.)
          sel.append(r)
    style='body{background:#090e19;color:#e5e7eb;font:14px system-ui;padding:18px}main{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}section{padding:12px;border:1px solid #374151;background:#1f2937;border-radius:8px}small{display:block;color:#bac7d5;margin:6px 0}svg{width:100%;max-width:330px;display:block;margin:auto}h1{font-size:22px}p{max-width:980px;line-height:1.5}'
    s=f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>DRIV3R C27 executed-trajectory comparison</title><style>{style}</style></head><body><h1>C27: deeper traps and premature STOP — actual executed trajectories</h1><p>Cells in grey are host-protected. White is the post-induction starting cursor, green is the current goal; strokes are executed legal actions. C18-A0 is the retained control. STOP suppression is a deliberately non-promotable diagnostic that alters cue-off behaviour; it is not a trained policy and does not prove recovery.</p><main>{"".join(panel(r) for r in sel)}</main></body></html>'
    Path(out).write_text(s)
    return len(sel)

if __name__=='__main__':
    import sys
    print('panels',generate(sys.argv[1],sys.argv[2]))