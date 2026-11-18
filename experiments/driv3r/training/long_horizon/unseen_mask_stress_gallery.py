"""Render actual executed neural and delayed-planner traces; no generated paths."""
import argparse, json
from pathlib import Path
from .unseen_mask_stress_lab import novel_world, MASKS, GOALS
from .world import Spec, MATERIAL_Z


def draw(run:Path,out:Path):
    docs=[]
    for arm in ('neural','absolute24'):
        for mask in MASKS:
            matches=sorted((run/'traces').glob(f'711-pickets-3000-{mask}-0-{arm}.json'))
            if len(matches)!=1:raise ValueError('Missing exact frozen research trace')
            docs.append(json.loads(matches[0].read_text()))
    txt=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 600" width="1120" height="600">',
         '<rect width="1120" height="600" fill="#101b24"/>',
         '<text x="20" y="25" font-family="monospace" font-size="15" fill="#fff">UNSEEN-MASK-STRESS-001 — executed synthetic DEVELOPMENT paths</text>']
    def esc(s):return str(s).replace('&','&amp;').replace('<','&lt;')
    for i,tr in enumerate(docs):
        row=tr['row'];x0=18+(i%4)*278;y0=42+(i//4)*275
        sp=Spec(**tr['spec']);w=novel_world(sp,row['layout'])
        txt.append(f'<g transform="translate({x0},{y0})">')
        txt.append('<rect x="0" y="0" width="265" height="265" rx="6" fill="#172a36" stroke="#48606f"/>')
        status='ARRIVED' if row['arrived_final'] else 'MISSED'
        txt.append(f'<text x="12" y="18" font-size="12" font-family="monospace" fill="#fff">{esc(row["layout"])} / {esc(row["arm"][:11])}</text>')
        txt.append(f'<text x="12" y="33" font-size="11" font-family="monospace" fill="{"#67e7b1" if row["arrived_final"] else "#ffad91"}">{status} · {row["decisions"]} decisions · {row["planner_calls"]} searches</text>')
        # Draw an exact pixel grid representing only the centre of the synthetic 96x96 world.
        scale=6.3;ox=31;oy=49
        def xy(p):return (ox+(p[0]-39)*scale,oy+(p[1]-37)*scale)
        for (cx,cy),mat in w.materials.items():
            if 39<=cx<=73 and 37<=cy<=71:
                x,y=xy((cx,cy));color='#7a91a4' if mat!=MATERIAL_Z else '#b17f68'
                txt.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{scale-0.4:.1f}" height="{scale-0.4:.1f}" fill="{color}"/>')
        for p in w.protected:
            if 39<=p[0]<=73 and 37<=p[1]<=71:
                x,y=xy(p);txt.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{scale-0.4:.1f}" height="{scale-0.4:.1f}" fill="#f06b78"/>')
        pts=[(48,48)]+[tuple(e['receipt']['cursor_after']) for e in tr['entries']]
        coords=[xy(p) for p in pts]
        line=' '.join(f'{x+scale/2:.1f},{y+scale/2:.1f}' for x,y in coords)
        txt.append(f'<polyline points="{line}" fill="none" stroke="#56d4ed" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>')
        for j,g in enumerate(GOALS):
            x,y=xy(g);txt.append(f'<circle cx="{x+scale/2:.1f}" cy="{y+scale/2:.1f}" r="3.8" fill="{"#f6ce54" if j==0 else "#b6a3ff"}"/>')
        txt.append('</g>')
    txt.append('<text x="20" y="592" font-size="11" font-family="monospace" fill="#b9cfda">orange = material, red = protected, turquoise = executed path, gold/purple = alternating targets; no native CyberSand receipts</text>')
    txt.append('</svg>')
    out.write_text('\n'.join(txt)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();draw(a.run,a.out)
