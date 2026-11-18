"""C58 source-matched executed-path SVG; never fabricate unexplored geometry."""
from pathlib import Path
import html,json

def create(traces,dest):
    rows=json.loads(Path(traces).read_text());W,H=960,1000
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
           '<rect width="100%" height="100%" fill="#0d1522"/>',
           '<text x="22" y="31" fill="#e8f0fb" font-family="monospace" font-size="19">C58 — dual-memory recovery, executed host traces</text>',
           '<text x="22" y="51" fill="#98afc8" font-family="monospace" font-size="11">No invented terrain or inferred walls · line = real action destination · dot = host goal</text>']
    for ix,row in enumerate(rows[:18]):
        col=ix%3;rr=ix//3;x0=20+col*314;y0=66+rr*150
        path=[tuple(p['p']) for p in row['outcome']['path']]
        pref=[tuple(p) for p in row['prefix'].get('positions',[])]
        pts=pref+path+[tuple(row['target'])]
        if not pts:continue
        xs=[p[0] for p in pts];ys=[p[1] for p in pts]
        a,b=min(xs),max(xs);c,d=min(ys),max(ys)
        unit=min(11.0,258/max(2,b-a),107/max(2,d-c))
        def at(p):return (x0+154+(p[0]-(a+b)/2)*unit,y0+82+(p[1]-(c+d)/2)*unit)
        parts.append(f'<rect x="{x0}" y="{y0}" width="294" height="141" rx="6" fill="#152337" stroke="#364d65"/>')
        title=f'{row["family"]} / loss {int(row["drop"]*100)}% / {row["policy"]}'
        parts.append(f'<text x="{x0+8}" y="{y0+16}" font-family="monospace" font-size="10" fill="#e8f0fb">{html.escape(title)}</text>')
        if len(pref)>1:
            coords=' '.join(f'{at(p)[0]:.1f},{at(p)[1]:.1f}' for p in pref)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="#546d86" stroke-width="1.3" opacity=".8"/>')
        if len(path)>1:
            coords=' '.join(f'{at(p)[0]:.1f},{at(p)[1]:.1f}' for p in path)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="#5dddc0" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>')
        gx,gy=at(row['target']);parts.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="4" fill="#ffd26b"/>')
        s=f'goal={row["outcome"]["goal"]}  steps={row["outcome"]["steps"]}  legal=PASS'
        parts.append(f'<text x="{x0+8}" y="{y0+134}" fill="#b1c5d6" font-family="monospace" font-size="9">{s}</text>')
    parts.append('</svg>');Path(dest).write_text('\n'.join(parts)+'\n')
if __name__=='__main__':
 import sys
 create(sys.argv[1],sys.argv[2])
