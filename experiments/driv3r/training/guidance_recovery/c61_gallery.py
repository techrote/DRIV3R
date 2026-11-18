"""Render only ACTUAL host-executed C61 movement segments; no invented map."""
from __future__ import annotations
import json,html,sys
from pathlib import Path
from collections import Counter


def build(traces,out):
    cols=4; rows=(len(traces)+cols-1)//cols;cw=255;ch=230
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {cols*cw} {rows*ch+45}" width="{cols*cw}" height="{rows*ch+45}">',
      '<rect width="100%" height="100%" fill="#0d1521"/>',
      '<text x="16" y="26" fill="#ecf2f9" font-family="sans-serif" font-size="18" font-weight="bold">DRIV3R C61 — executed remote-goal paths</text>',
      '<text x="16" y="42" fill="#8dabbc" font-family="sans-serif" font-size="10">Source-verified actions only; no inferred obstacles, passages or topology</text>']
    for i,tr in enumerate(traces):
        x=(i%cols)*cw;y=(i//cols)*ch+45
        pts=[tuple(tr['start_pos'])]+[tuple(v['p']) for v in tr['path']]
        xs=[p[0] for p in pts]+[tr['target'][0]];ys=[p[1] for p in pts]+[tr['target'][1]]
        lo_x,hi_x=min(xs)-2,max(xs)+2;lo_y,hi_y=min(ys)-2,max(ys)+2
        scale=min((cw-27)/max(1,hi_x-lo_x),(ch-52)/max(1,hi_y-lo_y))
        def xy(p):return f'{x+14+(p[0]-lo_x)*scale:.1f},{y+33+(p[1]-lo_y)*scale:.1f}'
        path=' '.join(xy(p) for p in pts)
        svg.extend([f'<rect x="{x+5}" y="{y+5}" width="{cw-10}" height="{ch-10}" rx="7" fill="#152233" stroke="#324359"/>',
          f'<text x="{x+15}" y="{y+21}" fill="#e2edfa" font-size="11" font-family="sans-serif">{html.escape(tr["family"])} · {html.escape(tr["policy"])} · erased {round(tr["drop"]*100)}%</text>',
          f'<polyline points="{path}" fill="none" stroke="{("#4ac69a" if tr["goal"] else "#efb55b")}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" opacity=".92"/>',
          f'<circle cx="{xy(tr["start_pos"]).split(",")[0]}" cy="{xy(tr["start_pos"]).split(",")[1]}" r="3" fill="#5bc0f4"/>',
          f'<circle cx="{xy(tr["target"]).split(",")[0]}" cy="{xy(tr["target"]).split(",")[1]}" r="4" fill="none" stroke="#ef6b88" stroke-width="1.5"/>',
          f'<text x="{x+15}" y="{y+ch-14}" fill="#9bb1c4" font-size="10" font-family="sans-serif">{tr["steps"]} executed actions · {"GOAL" if tr["goal"] else "not reached"} · {tr["graph"][:8]}</text>'])
    svg.append('</svg>')
    Path(out).write_text('\n'.join(svg)+'\n')
    print(f'gallery {out} {len(traces)} execution-matched paths')

if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit('usage: c61_gallery.py C61-traces.json output.svg')
    build(json.loads(Path(sys.argv[1]).read_text()),sys.argv[2])
