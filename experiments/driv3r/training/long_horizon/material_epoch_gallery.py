"""Original SVG gallery of replayed material-epoch action receipts (not native pixels)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .material_epoch_lab import LAYOUTS, PATCHES, GOALS
from .topology_adaptive_handoff_lab import overlay_world
from .world import Spec, MATERIAL_Z

def render(traces:Path,out:Path):
    width,height=1620,1880
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
       '<rect width="100%" height="100%" fill="#101725"/>',
       '<text x="44" y="55" font-family="sans-serif" fill="#f4f6f9" font-size="34" font-weight="bold">DRIV3R / material epoch — executed trajectories</text>',
       '<text x="44" y="84" font-family="sans-serif" fill="#aeb7c6" font-size="17">DEVELOPMENT synthetic state · 64 logical-move gate · fixed tick-24 versus material-epoch invalidation</text>']
    n=0
    for ri,layout in enumerate(LAYOUTS):
        for ci,arm in enumerate(('fixed24','fixed24_epoch')):
            matches=sorted(traces.glob(f'711-*-{layout}-0-{arm}.json'))
            assert matches, (layout,arm)
            tr=json.loads(matches[0].read_text());r=tr['row'];world=overlay_world(Spec(**tr['spec']),layout)
            tx=44+ci*790;ty=120+ri*425;sz=13
            s.append(f'<rect x="{tx-10}" y="{ty-20}" width="754" height="407" rx="14" fill="#1b2635" stroke="#384659"/>')
            s.append(f'<text x="{tx+8}" y="{ty+12}" fill="#f4f6f9" font-family="sans-serif" font-size="21">{layout.replace("_"," ").title()} · {arm}</text>')
            s.append(f'<text x="{tx+8}" y="{ty+36}" fill="#aeb7c6" font-family="monospace" font-size="13">final={r["arrived_final"]} · calls={r["planner_calls"]} · neural={r["neural_actions"]} · patches={r["material_applied"]}</text>')
            x0,y0=41,39; ox=tx+116;oy=ty+58
            s.append(f'<rect x="{ox}" y="{oy}" width="{31*sz}" height="{31*sz}" fill="#141e2b" stroke="#455369"/>')
            for (x,y),value in world.materials.items():
                if x0<=x<=71 and y0<=y<=69:
                    col='#758ba5' if value==MATERIAL_Z else '#5d6878'
                    s.append(f'<rect x="{ox+(x-x0)*sz}" y="{oy+(y-y0)*sz}" width="{sz-0.8}" height="{sz-0.8}" fill="{col}"/>')
            for (x,y) in world.protected:
                if x0<=x<=71 and y0<=y<=69:
                    s.append(f'<rect x="{ox+(x-x0)*sz}" y="{oy+(y-y0)*sz}" width="{sz-0.8}" height="{sz-0.8}" fill="#bd84c8"/>')
            def pos(p):return ox+(p[0]-x0+.5)*sz,oy+(p[1]-y0+.5)*sz
            for i,g in enumerate(GOALS):
                gx,gy=pos(g);s.append(f'<circle cx="{gx}" cy="{gy}" r="8.5" fill="none" stroke="{["#e7d07c","#82dab4"][i]}" stroke-width="3"/>')
            for ev in tr['material_events']:
                for p in ev['operations']:
                    ex,ey=pos(p['cell']); col='#ff8174' if p['op']=='close' else '#61d0df'
                    s.append(f'<circle cx="{ex}" cy="{ey}" r="5.8" fill="none" stroke="{col}" stroke-width="2"/>')
            for event in tr['entries']:
                before=event['receipt']['cursor_before'];after=event['receipt']['cursor_after']; bx,by=pos(before);ax,ay=pos(after)
                if before!=after:
                    col='#79c5ee' if event['source']=='neural' else '#e7ae76'
                    s.append(f'<line x1="{bx}" y1="{by}" x2="{ax}" y2="{ay}" stroke="{col}" stroke-width="2.5" opacity=".83" stroke-linecap="round"/>')
            sx,sy=pos((48,48));s.append(f'<circle cx="{sx}" cy="{sy}" r="5" fill="#f4e9c3"/>')
            n+=1
    s.append('<text x="44" y="1865" fill="#aeb7c6" font-family="sans-serif" font-size="14">Cyan paths: neural · amber paths: planner · outlined circles: externally scripted opening / closure · not native CyberSand footage</text>')
    s.append('</svg>')
    out.write_text('\n'.join(s)+'\n');return n

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--traces',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    x=a.parse_args();print('rendered_panels',render(x.traces,x.out))
