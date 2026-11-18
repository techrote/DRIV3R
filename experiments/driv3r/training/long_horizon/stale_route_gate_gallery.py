"""Draw original replayed synthetic trajectories; no drawn or invented paths."""
from __future__ import annotations
import argparse
from collections import defaultdict
from html import escape
import json
from pathlib import Path
from .stale_route_gate_lab import LAYOUTS, GOALS, scenario, verify_trace
from .world import Spec, MATERIAL_Z
from .topology_adaptive_handoff_lab import overlay_world
from .goal_conditioning_lab import GoalRanker

COLORS = {'stale_cache': '#727e8f', 'route_overlap': '#19a9cb',
          'lazy_illegal': '#54aa5c', 'neural_recovery': '#d99345'}
ORDER = list(COLORS)


def generate(run_dir: Path, checkpoints: Path, target: Path) -> None:
    models = {s: GoalRanker.load(checkpoints / f'{s}-conditioned.npz') for s in (711, 712)}
    traces = [json.loads(x) for x in (run_dir / 'receipts.jsonl').read_text().splitlines()]
    selected = {}
    for tr in traces:
        r = tr['row']
        if r['hazard'] == 'on_route' and r['arm'] in COLORS and r['model_seed'] in (None, 711):
            selected[(r['layout'], r['goal_idx'], r['arm'])] = tr
    expected = 4 * 2 * 4
    assert len(selected) == expected, (len(selected), expected)
    for tr in selected.values():
        verify_trace(tr, models)
    width, height = 990, 1710
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#101822"/>',
           '<text x="24" y="38" font-size="25" font-family="sans-serif" font-weight="bold" fill="#edf4ff">DRIV3R · Cached-route gate falsification</text>',
           '<text x="24" y="63" font-size="13" font-family="sans-serif" fill="#a8bfce">Actual synthetic DEVELOPMENT action receipts · one future-route material closure at tick 2 · 64-move gate</text>']
    for i, arm in enumerate(ORDER):
        x = 24 + i * 228
        out += [f'<line x1="{x}" y1="83" x2="{x+28}" y2="83" stroke="{COLORS[arm]}" stroke-width="4"/>',
                f'<text x="{x+36}" y="87" font-size="13" font-family="sans-serif" fill="#dce8f0">{escape(arm.replace("_", " "))}</text>']
    unit = 9.0
    for ri, layout in enumerate(LAYOUTS):
        for ci, gi in enumerate((0, 1)):
            x0, y0 = 23 + ci * 490, 116 + ri * 396
            out.append(f'<rect x="{x0}" y="{y0}" width="470" height="376" rx="10" fill="#182533" stroke="#375067"/>')
            out.append(f'<text x="{x0+14}" y="{y0+25}" font-family="sans-serif" font-size="16" fill="#eaf3ff">{escape(layout.replace("_", " "))} / goal {gi+1}</text>')
            tr = selected[(layout, gi, 'stale_cache')]
            sp = Spec(**tr['spec']); w = overlay_world(sp, layout)
            base_x, base_y = x0 + 58, y0 + 44
            def xy(point):
                return base_x + (point[0]-39)*unit, base_y + (point[1]-38)*unit
            out.append(f'<rect x="{base_x}" y="{base_y}" width="{33*unit}" height="{33*unit}" fill="#0b1620" stroke="#3b4d5e"/>')
            for p, kind in sorted(w.materials.items()):
                if 39 <= p[0] <= 71 and 38 <= p[1] <= 70:
                    x,y=xy(p)
                    out.append(f'<rect x="{x}" y="{y}" width="{unit}" height="{unit}" fill="{("#53636c" if kind != MATERIAL_Z else "#8d7796")}"/>')
            for p in w.protected:
                if 39 <= p[0] <= 71 and 38 <= p[1] <= 70:
                    x,y=xy(p)
                    out.append(f'<rect x="{x}" y="{y}" width="{unit}" height="{unit}" fill="#b5a16b"/>')
            for arm in ORDER:
                t = selected[(layout, gi, arm)]
                points = [(48,48)] + [tuple(e['receipt']['cursor_after']) for e in t['entries']]
                coords = ' '.join(f'{xy(p)[0]+unit/2:.1f},{xy(p)[1]+unit/2:.1f}' for p in points)
                out.append(f'<polyline points="{coords}" fill="none" stroke="{COLORS[arm]}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" opacity=".88"/>')
                out.append(f'<text x="{x0+358}" y="{y0+64+ORDER.index(arm)*19}" font-size="12" font-family="sans-serif" fill="{COLORS[arm]}">{escape(arm.split("_")[0])}: {"GOAL" if t["row"]["arrived"] else "FAIL"}</text>')
            bx,by=xy(tr['material_event']['cell'])
            out.append(f'<rect x="{bx-2}" y="{by-2}" width="{unit+4}" height="{unit+4}" fill="#f25257" stroke="#fff" stroke-width="1.4"/>')
            gx,gy=xy(GOALS[gi]);out.append(f'<circle cx="{gx+unit/2}" cy="{gy+unit/2}" r="6" fill="#f1d963" stroke="#25313c"/>')
            x,y=xy((48,48));out.append(f'<circle cx="{x+unit/2}" cy="{y+unit/2}" r="4" fill="#fff"/>')
            out.append(f'<text x="{x0+14}" y="{y0+359}" font-size="12" font-family="sans-serif" fill="#b8c8d4">Red square: new material · yellow circle: goal · white dot: start</text>')
    out.append('<text x="23" y="1696" font-size="12" fill="#8a9dad" font-family="sans-serif">Synthetic research paths, not native CyberSand game captures. No structural HELDOUT was opened.</text></svg>')
    target.write_text('\n'.join(out)+'\n')


def main():
    p = argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();generate(a.run,a.checkpoints,a.out)

if __name__ == '__main__':main()
