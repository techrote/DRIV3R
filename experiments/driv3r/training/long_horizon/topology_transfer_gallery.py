"""Render six exact-receipt research comparisons to a self-contained SVG.

No inferred path segments: every point comes from recorded cursor_after receipts.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import html
from .topology_transfer_lab import DEV_MOTIFS, fixture
from .world import Spec
from .train import sha_file

EXAMPLES=(
    (821,'ridges',3100,'lattice',3,3,0),
    (821,'islands',3200,'stagger',3,3,0),
    (821,'zigzag',3300,'stagger',3,3,0),
    (822,'pickets',3000,'stagger',3,3,1),
    (822,'ridges',3100,'stagger',3,3,1),
    (822,'zigzag',3300,'lattice',5,1,0),
)
COLORS={'new':'#2e9ea6','frozen':'#d06b45','prefix_oracle':'#3c465a'}


def find_trace(out, case, arm):
    seed,family,world_seed,motif,depth,quarter,gi=case
    f=out/'traces'/f'{seed}-{family}-{world_seed}-{motif}-{depth}-{quarter}-{gi}-{arm}.json'
    return json.loads(f.read_text())


def svg(out, destination):
    W,H=1320,890
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
      '<rect width="100%" height="100%" fill="#f7f7f4"/>',
      '<text x="28" y="36" fill="#18222e" font-size="25" font-family="sans-serif" font-weight="bold">DRIV3R / topology transfer — exact synthetic action receipts</text>',
      '<text x="28" y="59" fill="#536070" font-size="14" font-family="sans-serif">New train-only imitation vs frozen goal policy vs prefix-search control · development fixtures · NOT native CyberSand</text>']
    for i,case in enumerate(EXAMPLES):
        seed,family,world_seed,motif,depth,quarter,gi=case
        x0=24+(i%3)*435;y0=82+(i//3)*383
        traces={arm:find_trace(out,case,arm) for arm in COLORS}
        base=traces['new']['row'];sp=Spec(**traces['new']['spec']);world,goal,_=fixture(sp,motif,depth,quarter,gi)
        assert world.state_sha()==base['prelude_sha256']
        title=f'{family} · {motif} · seed {seed} · depth {depth}'
        parts += [f'<rect x="{x0}" y="{y0}" width="421" height="367" fill="white" stroke="#c8d0d4" rx="10"/>',
          f'<text x="{x0+15}" y="{y0+26}" fill="#202b35" font-size="17" font-family="sans-serif" font-weight="bold">{html.escape(title)}</text>']
        for arm in COLORS:
            r=traces[arm]['row'];st='GOAL' if r['arrived'] else 'FAIL'
            parts.append(f'<text x="{x0+15}" y="{y0+46+16*list(COLORS).index(arm)}" fill="{COLORS[arm]}" font-size="12" font-family="sans-serif">{arm}: {st}, {r["actions"]} actions</text>')
        # Local coordinate frame covers all visible trajectory endpoints while preserving true map geometry.
        vertices=[tuple(e['cursor_after']) for t in traces.values() for e in t['prelude']+t['events']]
        minx=max(0,min(x for x,y in vertices+[goal])-1);maxx=min(95,max(x for x,y in vertices+[goal])+1)
        miny=max(0,min(y for x,y in vertices+[goal])-1);maxy=min(95,max(y for x,y in vertices+[goal])+1)
        # Show at least the authored central motif, expanding for any actual path.
        minx=min(minx,38);maxx=max(maxx,64);miny=min(miny,38);maxy=max(maxy,64)
        scale=min(348/(maxx-minx+1),243/(maxy-miny+1))
        fx=lambda x:x0+42+(x-minx)*scale
        fy=lambda y:y0+100+(y-miny)*scale
        # Clip to intended frame by filtering positions, not clipping the trajectory itself.
        for (x,y),v in sorted(world.materials.items()):
            if minx<=x<=maxx and miny<=y<=maxy:
                parts.append(f'<rect x="{fx(x):.2f}" y="{fy(y):.2f}" width="{scale:.2f}" height="{scale:.2f}" fill="{ "#a0a5ac" if v==1 else "#777e8b" }"/>')
        for x,y in sorted(world.protected):
            if minx<=x<=maxx and miny<=y<=maxy:
                parts.append(f'<rect x="{fx(x):.2f}" y="{fy(y):.2f}" width="{scale:.2f}" height="{scale:.2f}" fill="#b84d60"/>')
        for arm in ('prefix_oracle','frozen','new'):
            trace=traces[arm]
            pts=[(48,48)]+[tuple(r['cursor_after']) for r in trace['prelude']+trace['events']]
            coords=' '.join(f'{fx(x)+scale/2:.1f},{fy(y)+scale/2:.1f}' for x,y in pts)
            dash=' stroke-dasharray="4 5"' if arm=='prefix_oracle' else ''
            parts.append(f'<polyline points="{coords}" stroke="{COLORS[arm]}" stroke-width="{2.7 if arm=="new" else 1.9}" fill="none" opacity=".85" stroke-linejoin="round" stroke-linecap="round"{dash}/>' )
            px,py=pts[-1]
            parts.append(f'<circle cx="{fx(px)+scale/2:.1f}" cy="{fy(py)+scale/2:.1f}" r="3.2" fill="{COLORS[arm]}"/>')
        parts.append(f'<circle cx="{fx(goal[0])+scale/2:.1f}" cy="{fy(goal[1])+scale/2:.1f}" r="5.3" fill="#f5ca4e" stroke="#4e3811" stroke-width="1.5"/>')
        parts.append(f'<circle cx="{fx(48)+scale/2:.1f}" cy="{fy(48)+scale/2:.1f}" r="3.5" fill="none" stroke="#151d2a" stroke-width="1.7"/>')
        parts.append(f'<text x="{x0+15}" y="{y0+353}" font-size="11" font-family="sans-serif" fill="#64717d">Goal: {goal} · material/protected cells shown · source: {str(base["base_sha256"])[:12]}</text>')
    parts.append('<text x="27" y="866" fill="#536070" font-size="13" font-family="sans-serif">Teal: new-trained  ·  Coral: frozen goal policy  ·  Gray dashed: deterministic prefix control  ·  Gold: goal  ·  Red: protected</text>')
    parts.append('</svg>')
    destination.write_text('\n'.join(parts)+'\n')
    return {'panels':len(EXAMPLES),'sha256':sha_file(destination),'bytes':destination.stat().st_size}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--svg',type=Path,required=True)
    a=p.parse_args();print(json.dumps(svg(a.out,a.svg),indent=2))

if __name__=='__main__':main()
