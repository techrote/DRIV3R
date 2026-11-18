"""Original path evidence SVG from verified synthetic action receipts."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
from .retarget_handoff_lab import fixture,ARMS,verify_trace
from .goal_conditioning_lab import GoalRanker
from .world import Spec

def render(archive,checkpoints,destination):
    traces=[json.loads(p.read_text()) for p in sorted((archive/'traces').glob('*.json'))]
    # Exact same paired start, world, goals and checkpoint across the six arms.
    examples=[t for t in traces if t['row']['family']=='pickets' and
              t['row']['model_seed']==711 and t['row']['layout']=='east_sleeve' and
              t['row']['goals']==[[60,48],[55,56]]]
    assert len(examples)==len(ARMS)
    lookup={t['row']['arm']:t for t in examples}; model=GoalRanker.load(checkpoints/'711-conditioned.npz')
    palette=['#efb153','#80a7d1','#8fcfa7','#dc8190','#ad8ed0','#c4d0df']
    W,H=1200,760; content=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
            '<rect width="100%" height="100%" fill="#101722"/>',
            '<text x="30" y="32" fill="white" font-family="sans-serif" font-size="19">RETARGET-HANDOFF-001 • executed synthetic receipts • goal changes at decision 8</text>']
    for i,arm in enumerate(ARMS):
        t=lookup[arm];verify_trace(t,model);row=t['row'];sp=Spec(**t['spec'])
        world=fixture(sp,5,row['layout'])
        events=t['prelude']+[e['receipt'] for e in t['entries']]
        pts=[tuple(events[0]['cursor_before'])]+[tuple(e['cursor_after']) for e in events]
        minx,maxx=min(p[0] for p in pts),max(p[0] for p in pts)
        miny,maxy=min(p[1] for p in pts),max(p[1] for p in pts)
        x0,x1,y0,y1=40,70,40,86
        assert x0<=minx<=maxx<=x1 and y0<=miny<=maxy<=y1
        xoff=24+(i%3)*396;yoff=53+(i//3)*349
        usable_w,usable_h=351,274;scale=min(usable_w/(x1-x0+1),usable_h/(y1-y0+1))
        cx=xoff+10;cy=yoff+43
        tx=lambda x:round(cx+(x-x0)*scale,2)
        ty=lambda y:round(cy+(y-y0)*scale,2)
        content.append(f'<rect x="{xoff}" y="{yoff}" width="379" height="338" rx="10" fill="#1b2736" stroke="#39485a"/>')
        content.append(f'<text x="{xoff+14}" y="{yoff+26}" font-family="sans-serif" font-size="16" fill="#ffffff">{html.escape(arm)} • {"ARRIVED" if row["arrived"] else row["reason"]} • {row["planner_calls"]} searches</text>')
        for (x,y),kind in world.materials.items():
            if x0<=x<=x1 and y0<=y<=y1:
                content.append(f'<rect x="{tx(x)}" y="{ty(y)}" width="{max(1,scale-.4):.2f}" height="{max(1,scale-.4):.2f}" fill="{"#5c6672" if kind==1 else "#855c77"}" opacity=".65"/>')
        for x,y in world.protected:
            if x0<=x<=x1 and y0<=y<=y1:
                content.append(f'<rect x="{tx(x)}" y="{ty(y)}" width="{max(1,scale-.4):.2f}" height="{max(1,scale-.4):.2f}" fill="#b03e5a"/>')
        line=' '.join(f'{tx(x)+scale/2},{ty(y)+scale/2}' for x,y in pts)
        content.append(f'<polyline points="{line}" fill="none" stroke="{palette[i]}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>')
        goal0,goal1=row['goals']
        for g,c in ((goal0,'#f9c956'),(goal1,'#77e0d4')):
            content.append(f'<circle cx="{tx(g[0])+scale/2}" cy="{ty(g[1])+scale/2}" r="5.4" fill="none" stroke="{c}" stroke-width="2.2"/>')
        switched=next(e for e in t['entries'] if e['tick']==8)
        sw=switched['receipt']['cursor_before']
        content.append(f'<circle cx="{tx(sw[0])+scale/2}" cy="{ty(sw[1])+scale/2}" r="4.6" fill="#fff"/>')
        content.append(f'<text x="{xoff+14}" y="{yoff+328}" font-family="sans-serif" font-size="12" fill="#dce8f2">{row["decisions"]} decisions • {row["neural_actions"]} neural • {row["logical_moves"]}/64 moves</text>')
    content.append('<text x="27" y="752" font-family="sans-serif" font-size="12" fill="#a8b7c8">Yellow ring: original goal • cyan ring: replacement goal • white dot: external switch • colour: executed history • grey/pink: materials/protection</text>')
    content.append('</svg>');destination.write_text('\n'.join(content)+'\n')
    return destination

def main():
    a=argparse.ArgumentParser();a.add_argument('--archive',type=Path,required=True);a.add_argument('--checkpoints',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    opts=a.parse_args();print(render(opts.archive,opts.checkpoints,opts.out))
if __name__=='__main__':main()
