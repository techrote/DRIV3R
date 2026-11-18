"""Render only measured, replayed DEVELOPMENT trace positions; no decorative paths."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
from .world import World, Spec
from .detour_recovery_lab import verify_trace
from .goal_conditioning_lab import GoalRanker

COLORS = {'conditioned':'#1aafa2','blind':'#b083e8','prefix_search':'#e3a847'}
X0,Y0,X1,Y1=43,37,71,65
CELL=10
WIDTH,HEIGHT=(X1-X0)*CELL,(Y1-Y0)*CELL


def plot(path: Path, trace_dir: Path, checkpoint_dir: Path):
    pieces=['<svg xmlns="http://www.w3.org/2000/svg" width="1152" height="840" viewBox="0 0 1152 840">',
            '<rect width="1152" height="840" fill="#0c1523"/>',
            '<text x="30" y="35" fill="#eef3f7" font-size="22" font-family="sans-serif" font-weight="bold">DRIV3R / DETOUR-RECOVERY-001</text>',
            '<text x="30" y="58" fill="#94a7ba" font-size="12" font-family="sans-serif">Actual DEV action receipts · fixed painted wrong-move prelude · 64 synthetic logical moves</text>']
    for row, goal_index in enumerate((0,1)):
        goal=((60,48),(55,56))[goal_index]
        for col, arm in enumerate(('conditioned','blind','prefix_search')):
            p=trace_dir/f'711-pickets-3000-5-{goal_index}-{arm}.json'
            trace=json.loads(p.read_text())
            model=GoalRanker.load(checkpoint_dir/'711-conditioned.npz') if arm=='conditioned' else (GoalRanker.load(checkpoint_dir/'711-blind.npz') if arm=='blind' else None)
            verify_trace(trace,model)
            r=trace['row']
            ox=28+col*380; oy=95+row*370
            pieces+= [f'<rect x="{ox}" y="{oy}" width="358" height="349" rx="12" fill="#17283b" stroke="#33516a"/>',
                      f'<text x="{ox+14}" y="{oy+24}" fill="#edf3f8" font-family="sans-serif" font-size="15" font-weight="bold">{html.escape(arm.replace("_"," ").title())} · goal {goal_index+1}</text>',
                      f'<text x="{ox+14}" y="{oy+44}" fill="#b5c4d5" font-family="sans-serif" font-size="11">arrival {"YES" if r["arrived"] else "NO"} · moves {r["total_logical_moves"]}/64 · recovery {r["recovery_actions"]}</text>']
            px=ox+38;py=oy+56
            pieces.append(f'<defs><clipPath id="clip-{row}-{col}"><rect x="{px}" y="{py}" width="{WIDTH}" height="{HEIGHT}"/></clipPath></defs>')
            pieces.append(f'<g clip-path="url(#clip-{row}-{col})"><rect x="{px}" y="{py}" width="{WIDTH}" height="{HEIGHT}" fill="#101d2c"/>')
            w=World.create(Spec(**trace['spec']))
            for (x,y),mat in w.materials.items():
                if X0<=x<X1 and Y0<=y<Y1:
                    pieces.append(f'<rect x="{px+(x-X0)*CELL}" y="{py+(y-Y0)*CELL}" width="{CELL}" height="{CELL}" fill="{ "#5e6d7b" if mat==1 else "#916c58" }"/>')
            for x,y in w.protected:
                if X0<=x<X1 and Y0<=y<Y1:
                    pieces.append(f'<rect x="{px+(x-X0)*CELL}" y="{py+(y-Y0)*CELL}" width="{CELL}" height="{CELL}" fill="#e16c6c"/>')
            events=trace['prelude']+trace['events']
            visited=[tuple(events[0]['cursor_before'])]+[tuple(e['cursor_after']) for e in events]
            def at(pt): return f'{px+(pt[0]-X0+.5)*CELL:.1f},{py+(pt[1]-Y0+.5)*CELL:.1f}'
            prelude=[tuple(events[0]['cursor_before'])]+[tuple(e['cursor_after']) for e in trace['prelude']]
            pieces.append(f'<polyline points="{" ".join(at(v) for v in prelude)}" stroke="#ffdb88" stroke-width="3.7" stroke-dasharray="5,3" fill="none" opacity="0.9"/>')
            cont=visited[len(trace['prelude']):]
            pieces.append(f'<polyline points="{" ".join(at(v) for v in cont)}" stroke="{COLORS[arm]}" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" fill="none"/>')
            gx,gy=(px+(goal[0]-X0+.5)*CELL,py+(goal[1]-Y0+.5)*CELL)
            pieces.append(f'<polygon points="{gx},{gy-7} {gx+7},{gy} {gx},{gy+7} {gx-7},{gy}" stroke="#ffffff" stroke-width="1.5" fill="#ffe87a"/>')
            for pt,color,rad in [(visited[0],'#ffffff',4),(visited[-1],COLORS[arm],4)]:
                x,y=map(float,at(pt).split(','));pieces.append(f'<circle cx="{x}" cy="{y}" r="{rad}" fill="{color}" stroke="#0b1020" stroke-width="1"/>')
            pieces.append('</g>')
            pieces.append(f'<text x="{ox+14}" y="{oy+341}" fill="#92b0c3" font-family="sans-serif" font-size="11">{len(trace["events"])} post-error decisions · target ({goal[0]}, {goal[1]})</text>')
    pieces.append('<text x="30" y="827" fill="#97a7b8" font-family="sans-serif" font-size="11">Orange dashes: identical 5-step forced eastward mistake · blue/grey: source materials · red: protected · diamond: required goal</text>')
    pieces.append('</svg>')
    path.write_text('\n'.join(pieces)+'\n')
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--traces',type=Path,required=True);p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); print(plot(args.output,args.traces,args.checkpoints))
