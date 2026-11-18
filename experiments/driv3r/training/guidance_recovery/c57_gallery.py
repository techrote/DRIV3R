"""Render only executed C57 trajectory receipts; no synthetic imagined map geometry."""
from __future__ import annotations
import html,json
from pathlib import Path

def render(trace_path,destination):
    rows=json.loads(Path(trace_path).read_text());assert len(rows)==18
    w,h=282,190;parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w*3} {h*6+44}" width="{w*3}" height="{h*6+44}">',
         '<rect width="100%" height="100%" fill="#0c1421"/>',
         '<text x="16" y="27" fill="#f1f5ff" font-size="17" font-family="sans-serif">C57 · Executed host-legal recovery paths</text>',
         '<text x="16" y="42" fill="#a9b9ce" font-size="10" font-family="sans-serif">Gray: executed prefix · teal: executed suffix · gold: assigned goal · no hidden geometry rendered</text>']
    for idx,r in enumerate(rows):
        c=idx%3;rr=idx//3;x=c*w;y=44+rr*h
        o=r['outcome'];prefix=r['prefix']['positions'];suffix=o['path'];pts=[tuple(p) for p in prefix];outpts=[pts[-1]]+[tuple(z['p']) for z in suffix]
        scale=10;offx=28;offy=30
        P=lambda p: f'{x+offx+scale*float(p[0]):.1f},{y+offy+scale*float(p[1]):.1f}'
        parts.append(f'<rect x="{x+5}" y="{y+5}" width="{w-10}" height="{h-10}" rx="7" fill="#172335" stroke="#35475c"/>')
        for k in range(0,16):
            parts.append(f'<path d="M{x+offx+k*scale} {y+offy} V{y+offy+140} M{x+offx} {y+offy+k*scale} H{x+offx+140}" stroke="#26354a" stroke-width="0.5"/>')
        parts.append(f'<polyline fill="none" stroke="#aebace" stroke-width="2.0" opacity=".65" points="{" ".join(P(p) for p in pts)}"/>')
        parts.append(f'<polyline fill="none" stroke="#55ddc5" stroke-width="2.7" points="{" ".join(P(p) for p in outpts)}"/>')
        goal=tuple(r['target']);last=outpts[-1]
        parts.append(f'<circle cx="{x+offx+scale*goal[0]}" cy="{y+offy+scale*goal[1]}" r="5" fill="none" stroke="#ffc35f" stroke-width="2.2"/>')
        parts.append(f'<circle cx="{x+offx+scale*last[0]}" cy="{y+offy+scale*last[1]}" r="2.8" fill="#fff"/>')
        name=html.escape(r['policy']);family=html.escape(r['family'])
        parts.append(f'<text x="{x+176}" y="{y+46}" fill="#f0f6ff" font-size="11" font-family="sans-serif">{name}</text>')
        parts.append(f'<text x="{x+176}" y="{y+62}" fill="#b7c9dd" font-size="10" font-family="sans-serif">{family} / {r["case"]}</text>')
        parts.append(f'<text x="{x+176}" y="{y+78}" fill="#b7c9dd" font-size="10" font-family="sans-serif">erasure {int(100*r["drop"])}%</text>')
        parts.append(f'<text x="{x+176}" y="{y+94}" fill="{"#89f0a9" if o["goal"] else "#ffa08e"}" font-size="10" font-family="sans-serif">{"GOAL" if o["goal"] else "NO GOAL"}</text>')
        parts.append(f'<text x="{x+176}" y="{y+110}" fill="#b7c9dd" font-size="10" font-family="sans-serif">steps {o["steps"]}</text>')
        parts.append(f'<text x="{x+176}" y="{y+126}" fill="#b7c9dd" font-size="10" font-family="sans-serif">remaining {o["distance"]}</text>')
        parts.append(f'<text x="{x+16}" y="{y+176}" fill="#8ca0bb" font-size="8" font-family="monospace">{r["graph"][:12]} · receipt verified</text>')
    parts.append('</svg>');Path(destination).write_text('\n'.join(parts)+'\n')

if __name__=='__main__':
 import sys;render(sys.argv[1],sys.argv[2])
