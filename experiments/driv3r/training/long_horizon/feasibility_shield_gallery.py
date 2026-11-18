"""Generate source-faithful SVG from exact archived synthetic action receipts."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
from .detour_recovery_lab import fixture
from .world import Spec

PALETTE = {'neural':'#c04b52','late24':'#b98422','shield0':'#277e9a','planner':'#424a56'}
CASES = ((711,'pickets',3000,3,1),(711,'islands',3200,3,0),
         (711,'zigzag',3300,5,0),(712,'ridges',3100,3,1),
         (712,'islands',3200,5,0),(712,'zigzag',3300,5,1))


def draw(archive:Path,destination:Path):
    W,H=1380,920;panel_w=430;panel_h=416
    a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       '<rect width="100%" height="100%" fill="#f7f7f5"/>',
       '<text x="28" y="36" font-size="24" font-family="sans-serif" font-weight="bold" fill="#202833">DRIV3R — feasibility shield / executed paths</text>',
       '<text x="28" y="58" font-size="14" font-family="sans-serif" fill="#444">DEVELOPMENT-only synthetic host-legal action receipts · trajectories not native CyberSand</text>']
    for index,(seed,family,wseed,depth,goalid) in enumerate(CASES):
        root=archive/'traces'; items={}
        for arm in PALETTE:
            p=root/f'{seed}-{family}-{wseed}-{depth}-{goalid}-{arm}.json'
            items[arm]=json.loads(p.read_text())
        anytrace=items['neural'];sp=Spec(**anytrace['spec']);w=fixture(sp,depth)
        paths={}
        allpoints=[tuple(w.cursor),tuple(anytrace['row']['goal'])]
        for arm,t in items.items():
            pts=[tuple(w.cursor)]+[tuple(e['receipt']['cursor_after']) for e in t['entries']]
            paths[arm]=pts;allpoints+=pts
        lo_x=max(0,min(x for x,y in allpoints)-2);hi_x=min(95,max(x for x,y in allpoints)+2)
        lo_y=max(0,min(y for x,y in allpoints)-2);hi_y=min(95,max(y for x,y in allpoints)+2)
        # No clipped segments: every receipt position determines the viewport.
        cx=30+(index%3)*452;cy=92+(index//3)*424
        a.append(f'<rect x="{cx}" y="{cy}" width="{panel_w}" height="{panel_h}" fill="white" rx="9" stroke="#c5ccd3"/>')
        a.append(f'<text x="{cx+15}" y="{cy+25}" font-family="sans-serif" font-size="16" font-weight="bold" fill="#253342">{family} / {seed} / prelude {depth} / goal {goalid+1}</text>')
        for j,arm in enumerate(PALETTE):
            row=items[arm]['row'];tag='GOAL' if row['arrived'] else 'FAIL'
            a.append(f'<text x="{cx+15+215*(j%2)}" y="{cy+49+19*(j//2)}" font-family="sans-serif" font-size="12" fill="{PALETTE[arm]}">{arm}: {tag} ({row["decisions"]})</text>')
        wx=cx+20;wy=cy+98; bw=392;bh=299
        scale=min(bw/max(1,hi_x-lo_x+1),bh/max(1,hi_y-lo_y+1),14)
        ox=wx+(bw-scale*(hi_x-lo_x+1))/2;oy=wy+(bh-scale*(hi_y-lo_y+1))/2
        def pos(pt):return ox+(pt[0]-lo_x+.5)*scale,oy+(pt[1]-lo_y+.5)*scale
        a.append(f'<rect x="{wx}" y="{wy}" width="{bw}" height="{bh}" fill="#f1f3f1"/>')
        for (x,y),v in sorted(w.materials.items()):
            if lo_x<=x<=hi_x and lo_y<=y<=hi_y:
                a.append(f'<rect x="{ox+(x-lo_x)*scale:.2f}" y="{oy+(y-lo_y)*scale:.2f}" width="{scale:.2f}" height="{scale:.2f}" fill="{"#9aa0a5" if v==1 else "#737e8b"}"/>')
        for x,y in sorted(w.protected):
            if lo_x<=x<=hi_x and lo_y<=y<=hi_y:
                a.append(f'<rect x="{ox+(x-lo_x)*scale:.2f}" y="{oy+(y-lo_y)*scale:.2f}" width="{scale:.2f}" height="{scale:.2f}" fill="#aa887e"/>')
        for arm,pts in paths.items():
            pstr=' '.join(f'{x:.2f},{y:.2f}' for x,y in map(pos,pts))
            a.append(f'<polyline points="{pstr}" fill="none" stroke="{PALETTE[arm]}" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" opacity=".79"/>')
        gx,gy=pos(tuple(anytrace['row']['goal']));sx,sy=pos((48,48))
        a.append(f'<circle cx="{gx:.2f}" cy="{gy:.2f}" r="5" fill="#22ad65" stroke="white" stroke-width="1.5"/>')
        a.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="4.5" fill="white" stroke="#303744" stroke-width="2"/>')
        a.append(f'<text x="{cx+15}" y="{cy+408}" font-family="sans-serif" font-size="11" fill="#555">view {lo_x}..{hi_x} × {lo_y}..{hi_y} · full path bounds · no interpolated points</text>')
    a.append('</svg>')
    destination.write_text('\n'.join(a)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    draw(args.archive,args.out)
