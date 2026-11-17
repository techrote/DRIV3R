"""C46 host-executed path exhibit, not a native CyberSand capture."""
import argparse,html,json
from pathlib import Path


def make(source,dest):
    records=json.loads(Path(source).read_text())
    blocks=[]
    for r in records:
        walls=set(tuple(p) for p in r['protected'])
        pts=[tuple(r['initial'])]+[tuple(p['position']) for p in r['trajectory']]
        frame=['<svg viewBox="0 0 240 240" role="img" aria-label="Synthetic host executed path">',
               '<rect width="240" height="240" fill="#101821"/>']
        for x,y in walls:frame.append(f'<rect x="{x*16}" y="{y*16}" width="16" height="16" fill="#3a4857"/>')
        line=' '.join(f'{x*16+8},{y*16+8}' for x,y in pts)
        frame.append(f'<polyline points="{line}" fill="none" stroke="#f0be6a" stroke-width="2.5" stroke-linejoin="round"/>')
        frame.append(f'<circle cx="{pts[0][0]*16+8}" cy="{pts[0][1]*16+8}" r="5" fill="#8addbf"/>')
        frame.append(f'<circle cx="{pts[-1][0]*16+8}" cy="{pts[-1][1]*16+8}" r="5" fill="#f17a87"/>')
        frame.append('</svg>')
        label=html.escape(r['model'])
        blocks.append(f'<article><h2>{label}</h2><p>Selected: {html.escape(r["selected"])} · Goals: {r["goals"]} · Steps: {len(r["trajectory"])} · Graph: {r["graph"][:10]}</p>{"".join(frame)}</article>')
    doc='''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>C46 host-executed trajectories</title>
<style>body{background:#101821;color:#f2f4f7;font:15px system-ui,sans-serif;margin:30px auto;max-width:1150px;padding:0 12px}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:16px}article{border:1px solid #45515e;border-radius:9px;padding:12px;background:#1c2832}article svg{width:100%;height:auto}h1{font-size:24px}h2{font-size:17px}p{color:#b7c7d1;font-size:12px}</style></head><body><h1>C46 — complete host-executed trajectories</h1><p>Observed paths, not target-oracle paths; synthetic research fixture only. Green start, red finish. Each panel is independently replayed against recorded hashes.</p><section>'''+''.join(blocks)+'</section></body></html>'
    Path(dest).write_text(doc)
    return len(records)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('traces');p.add_argument('html');a=p.parse_args();print(make(a.traces,a.html))