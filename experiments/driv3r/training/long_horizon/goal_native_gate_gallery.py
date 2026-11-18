"""SVG rendering of original executed action receipts, never native screenshots."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape
from .world import Spec, World

WIDTH, HEIGHT = 1092, 725
COLS = 3
ROWS = 2


def render(evidence_dir: Path, destination: Path):
    panels = [
        ('711-pickets-3000-0-conditioned.json', 'Pickets / conditioned'),
        ('711-pickets-3000-0-blind.json', 'Pickets / goal-blind'),
        ('711-pickets-3000-0-goal_oracle.json', 'Pickets / BFS-stack'),
        ('712-zigzag-3302-1-conditioned.json', 'Zigzag / conditioned / perturbed'),
        ('712-zigzag-3302-1-blind.json', 'Zigzag / goal-blind / perturbed'),
        ('712-zigzag-3302-1-goal_oracle.json', 'Zigzag / BFS-stack / perturbed'),
    ]
    # All panel selectors are exact original DEVELOPMENT traces; fail closed on missing.
    bits = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
            '<rect width="100%" height="100%" fill="#f8fafc"/>',
            '<text x="27" y="32" fill="#0f172a" font-family="sans-serif" font-size="19" font-weight="bold">GOAL-NATIVE-GATE-001 · actual synthetic action receipts</text>',
            '<text x="27" y="53" fill="#475569" font-family="sans-serif" font-size="12">Goal waypoints (orange) · executed path (teal) · blocked/protected cells (grey/red) · no native execution</text>']
    for k,(name,title) in enumerate(panels):
        trace = json.loads((evidence_dir/'traces'/name).read_text())
        s=trace['spec'];sp=Spec(**s)
        w=World.create(sp)
        r=trace['result']; rr=k//COLS; cc=k%COLS
        ox=20+cc*361; oy=68+rr*326
        bits.append(f'<g transform="translate({ox},{oy})">')
        bits.append('<rect width="345" height="311" rx="7" fill="white" stroke="#cbd5e1"/>')
        bits.append(f'<text x="13" y="23" font-family="sans-serif" font-size="14" font-weight="bold" fill="#0f172a">{escape(title)}</text>')
        bits.append(f'<text x="13" y="43" font-family="sans-serif" font-size="11" fill="#475569">{r["goals_reached"]}/{r["goals_offered"]} goals · {r["logical_moves"]} logical moves · {r["stop"]}</text>')
        # Unzoomed side-view of a centered 70x70 source window; no invented geometry.
        x0,y0=14,55; unit=3.20
        bits.append(f'<rect x="{x0}" y="{y0}" width="307" height="237" fill="#f1f5f9"/>')
        for (x,y),mat in w.materials.items():
            if 0<=x<96 and 11<=y<85:
                bits.append(f'<rect x="{x0+x*unit:.2f}" y="{y0+(y-11)*unit:.2f}" width="{unit:.2f}" height="{unit:.2f}" fill="#64748b"/>')
        for x,y in w.protected:
            if 11<=y<85:
                bits.append(f'<rect x="{x0+x*unit:.2f}" y="{y0+(y-11)*unit:.2f}" width="{unit:.2f}" height="{unit:.2f}" fill="#e11d48"/>')
        cursor=[(48,48)] + [tuple(e['cursor_after']) for e in trace['events']]
        coords=' '.join(f'{x0+x*unit+unit/2:.1f},{y0+(y-11)*unit+unit/2:.1f}' for x,y in cursor)
        bits.append(f'<polyline fill="none" stroke="#0891b2" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" points="{coords}"/>')
        for j,(x,y) in enumerate(r['targets']):
            if 11<=y<85:
                cx=x0+x*unit+unit/2;cy=y0+(y-11)*unit+unit/2
                bits.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.2" fill="#f59e0b" stroke="#ffffff" stroke-width="0.7"/>')
        x,y=cursor[-1];bits.append(f'<circle cx="{x0+x*unit+unit/2:.1f}" cy="{y0+(y-11)*unit+unit/2:.1f}" r="3.8" fill="#0f172a"/>')
        bits.append('</g>')
    bits.append('</svg>')
    destination.write_text('\n'.join(bits)+'\n')
    return {'panels': len(panels),'source_files': [p[0] for p in panels]}


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    print(json.dumps(render(a.evidence,a.out),sort_keys=True))
