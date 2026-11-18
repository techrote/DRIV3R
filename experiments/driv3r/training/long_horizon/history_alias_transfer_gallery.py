"""Deterministic exact-path SVG gallery for HISTORY-ALIAS-TRANSFER-001."""
from __future__ import annotations
import argparse, html
from pathlib import Path
from . import history_alias_transfer_lab as t
from .world import digest

def render(path):
    samples=[c for c in t.cases() if c['rotation'] in (90,180) and c['remaining'] in (2,3,4)]
    # Three budgets by two target conditions; all six from a single fixed motif.
    chosen=[next(c for c in samples if c['rotation']==90 and c['remaining']==b and c['kind']==k)
            for b in (2,3,4) for k in ('positive','negative')]
    tile_w,tile_h=380,320
    s=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 960" width="760" height="960">',
       '<rect width="760" height="960" fill="#f7f8fc"/>',
       '<text x="20" y="27" font-family="sans-serif" font-size="15" font-weight="bold">HISTORY-ALIAS-TRANSFER-001 • actual host-legal synthetic paths</text>']
    for i,c in enumerate(chosen):
        x0=(i%2)*tile_w;y0=(i//2)*tile_h+28
        w=c['world'];p=w.path;goal=c['goal']
        origin_x,origin_y=190,170
        def xy(v):return x0+origin_x+(v[0]-48)*18,y0+origin_y+(v[1]-48)*18
        label=f"remaining={c['remaining']} | {c['kind']} | expected={c['label']} | depth={c['depth']}"
        s.append(f'<g><rect x="{x0+8}" y="{y0+9}" width="364" height="292" rx="10" fill="white" stroke="#cdd3df"/>')
        s.append(f'<text x="{x0+20}" y="{y0+35}" font-size="12" font-family="monospace">{html.escape(label)}</text>')
        for v in sorted(w.ink):
            x,y=xy(v)
            s.append(f'<rect x="{x-5}" y="{y-5}" width="10" height="10" fill="#becbde"/>')
        seq=' '.join(f'{x:.2f},{y:.2f}' for x,y in map(xy,p))
        s.append(f'<polyline points="{seq}" fill="none" stroke="#1b5a96" stroke-width="2.3" stroke-linejoin="round"/>')
        for j,v in enumerate(p):
            x,y=xy(v)
            s.append(f'<circle cx="{x}" cy="{y}" r="2.4" fill="#164e85"/>')
        sx,sy=xy(w.cursor);gx,gy=xy(goal)
        s.append(f'<circle cx="{sx}" cy="{sy}" r="7" fill="#de7543" stroke="#fff" stroke-width="1.5"/>')
        s.append(f'<circle cx="{gx}" cy="{gy}" r="8" fill="none" stroke="#1a9973" stroke-width="3"/>')
        s.append(f'<text x="{x0+20}" y="{y0+284}" font-size="11" fill="#5d6880" font-family="sans-serif">Blue: history • Orange: cursor • Green: historical goal</text></g>')
    s.append('</svg>')
    Path(path).write_text('\n'.join(s)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args();render(args.output)
