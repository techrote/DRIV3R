"""Render the original 128-decision synthetic receipts; no invented waypoints."""
from __future__ import annotations
import argparse
import gzip
import html
import json
from pathlib import Path


def render(receipts: Path, destination: Path):
    chosen=[]
    with gzip.open(receipts,'rt') as z:
        for line in z:
            r=json.loads(line)
            if (r['arm']=='memory' and r['depth']==128 and
                r['spec']['seed']%100==0):
                chosen.append(r)
    assert len(chosen)==8
    chosen.sort(key=lambda r:(r['spec']['family'],r['side']))
    out=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 1710">',
         '<rect width="1200" height="1710" fill="#101c29"/>',
         '<text x="35" y="48" fill="#f2f6fb" font-size="32" font-family="sans-serif" font-weight="700">DRIV3R / INK-HISTORY ALIAS</text>',
         '<text x="35" y="81" fill="#b9d1df" font-size="17" font-family="sans-serif">Executed 128-action prelude · 2 earlier ink routes · identical recent observation</text>',
         '<text x="35" y="107" fill="#fbbf76" font-size="15" font-family="sans-serif">Synthetic DEVELOPMENT only; long preludes exceed historical native logical-move limits</text>']
    for i,r in enumerate(chosen):
        x=34+(i%2)*576;y=136+(i//2)*383
        events=r['events']; pre=events[:128]; post=events[128:]
        title=f"{r['spec']['family']} / earlier {r['side']} route"
        out+= [f'<rect x="{x}" y="{y}" width="556" height="362" rx="12" fill="#172d3b" stroke="#426478"/>',
              f'<text x="{x+18}" y="{y+34}" fill="#f1f6fa" font-size="21" font-family="sans-serif" font-weight="600">{html.escape(title)}</text>',
              f'<text x="{x+18}" y="{y+61}" fill="#a5bbca" font-size="15" font-family="sans-serif">128 host-menu actions · 248 synthetic logical moves</text>']
        ox=x+17;oy=y+83;s=2.65
        out.append(f'<rect x="{ox}" y="{oy}" width="254.4" height="254.4" fill="#0c1a23" stroke="#3f5864"/>')
        def coord(p):return (ox+float(p[0])*s,oy+float(p[1])*s)
        # Provenance-resolved actual executed MOVE receipts, with relocation breaks.
        segments=[];segment=[[48,48]]
        for e in pre:
            if e['chosen']=='RELOCATE_START':
                segments.append(segment);segment=[e['cursor_after']]
            else:segment.append(e['cursor_after'])
        segments.append(segment)
        for seg in segments:
            if len(seg)>1:
                pts=' '.join(f'{coord(p)[0]:.2f},{coord(p)[1]:.2f}' for p in seg)
                out.append(f'<polyline points="{pts}" fill="none" stroke="#64b6cb" stroke-width="2.4" opacity="0.8"/>')
        for e in post:
            if e['chosen'].startswith('MOVE_'):
                a,b=coord(e['cursor_before']),coord(e['cursor_after'])
                out.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="#ffc66c" stroke-width="3"/>')
        for cell,color in [((48,54),'#ffda6d'),((48,44),'#77e4b6')]:
            px,py=coord(cell);out.append(f'<circle cx="{px}" cy="{py}" r="5" fill="{color}"/>')
        # Local first decision: side routes differ only in old ink.
        fx=x+290;fy=y+94
        out.append(f'<text x="{fx}" y="{fy}" fill="#a5bbca" font-size="16" font-family="sans-serif">Same endpoint (48,54)</text>')
        out.append(f'<text x="{fx}" y="{fy+30}" fill="#a5bbca" font-size="16" font-family="sans-serif">Same legal action menu</text>')
        out.append(f'<text x="{fx}" y="{fy+60}" fill="#a5bbca" font-size="16" font-family="sans-serif">Same last 8 actions</text>')
        out.append(f'<text x="{fx}" y="{fy+107}" fill="#fbbf76" font-size="23" font-weight="600" font-family="sans-serif">First: {r["chosen"]}</text>')
        out.append(f'<text x="{fx}" y="{fy+140}" fill="#83e4bc" font-size="16" font-family="sans-serif">Shortest-path label: {r["expected_first_action"]}</text>')
        out.append(f'<text x="{fx}" y="{fy+173}" fill="#d5e2e7" font-size="15" font-family="sans-serif">Model probability:</text>')
        out.append(f'<text x="{fx}" y="{fy+197}" fill="#d5e2e7" font-size="15" font-family="monospace">{r["probs"].get(r["chosen"],0):.6f}</text>')
        out.append(f'<text x="{fx}" y="{fy+240}" fill="#a5bbca" font-size="14" font-family="sans-serif">Amber: planned suffix</text>')
        out.append(f'<text x="{fx}" y="{fy+262}" fill="#a5bbca" font-size="14" font-family="sans-serif">Cyan: actual ink trail</text>')
    out+=['<text x="35" y="1684" fill="#b9d1df" font-size="15" font-family="sans-serif">Coordinates depict synthetic World action receipts; route after first model decision is BFS-controlled, not model autonomy.</text>','</svg>']
    destination.write_text('\\n'.join(out)+'\\n')
    return len(chosen)

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--receipts',required=True,type=Path)
    q.add_argument('--out',required=True,type=Path);a=q.parse_args();print(render(a.receipts,a.out))
