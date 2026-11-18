"""Exact-receipt, full-history SVG comparisons; static data plot, no image generator."""
from __future__ import annotations
import argparse,html,json
from pathlib import Path
from .topology_goal_churn_lab import overlay_world,LAYOUTS
from .world import Spec

PALETTE=('#0f766e','#c084fc','#ed9232')
W,H=1070,1830
M=12
X0,Y0=35,30
SIZE=45
S=8.1

def esc(v):return html.escape(str(v),quote=True)

def render(root,out):
    candidates=[]
    for layout in LAYOUTS:
        traces=[json.loads(f.read_text()) for f in (root/'traces').glob(f'*{layout}*.json')]
        by={}
        for t in traces:
            r=t['row'];key=(r['world_family'],r['world_seed'],r['order'],r['model_seed']);by.setdefault(key,{})[r['arm']]=t
        matched=[(key,v) for key,v in by.items() if 'neural' in v and 'absolute24' in v
                 and not v['neural']['row']['arrived_final'] and v['absolute24']['row']['arrived_final']]
        if not matched: matched=[(k,v) for k,v in by.items() if 'neural' in v and 'absolute24' in v]
        if not matched: raise ValueError(f'No matching traces: {layout}')
        key,pair=sorted(matched,key=lambda x:x[0])[0]
        candidates.append((layout,key,pair))
    out.parent.mkdir(parents=True,exist_ok=True)
    ss=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="#0d1728"/>',
        '<text x="26" y="39" fill="#f8fafc" font-family="monospace" font-size="24" font-weight="bold">TOPOLOGY-GOAL-CHURN-001 · executed synthetic routes</text>',
        '<text x="26" y="64" fill="#a8b7ce" font-family="monospace" font-size="13">Identical state at start per paired arm · phase colors: 0 teal / 1 lilac / 2 amber</text>']
    cells=S
    for row_i,(layout,key,pair) in enumerate(candidates):
        for j,arm in enumerate(('neural','absolute24')):
            t=pair[arm];r=t['row'];spec=Spec(**t['spec']);world=overlay_world(spec,layout)
            ox=24+j*535;oy=90+row_i*430
            ss.append(f'<rect x="{ox}" y="{oy}" width="520" height="412" rx="10" fill="#14243b" stroke="#455c77"/>')
            ss.append(f'<text x="{ox+12}" y="{oy+24}" fill="#e2e8f0" font-family="monospace" font-size="17">{esc(layout)} · {esc(arm)}</text>')
            ss.append(f'<text x="{ox+12}" y="{oy+44}" fill="#a9b7c9" font-family="monospace" font-size="12">{esc(key[0])} seed {key[1]}/{key[3]} | goal {"PASS" if r["arrived_final"] else "FAIL"} | {r["decisions"]} actions | {r["planner_calls"]} searches</text>')
            px=ox+72;py=oy+55
            ss.append(f'<rect x="{px}" y="{py}" width="{SIZE*cells:.1f}" height="{SIZE*cells:.1f}" fill="#0e1c2e" stroke="#39536e"/>')
            for p,v in world.materials.items():
                if X0<=p[0]<X0+SIZE and Y0<=p[1]<Y0+SIZE:
                    x=px+(p[0]-X0)*cells;y=py+(p[1]-Y0)*cells
                    color = "#7d95b0" if v==1 else "#7c527b"
                    ss.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cells:.1f}" height="{cells:.1f}" fill="{color}" opacity="0.88"/>')
            for p in world.protected:
                if X0<=p[0]<X0+SIZE and Y0<=p[1]<Y0+SIZE:
                    x=px+(p[0]-X0)*cells;y=py+(p[1]-Y0)*cells
                    ss.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cells:.1f}" height="{cells:.1f}" fill="#e879a4" opacity="0.93"/>')
            groups=[[],[],[]];initial=[48,48]
            for e in t['entries']:
                dest=e['receipt']['cursor_after'];phase=e['phase'];groups[phase].append((dest[0],dest[1]))
            # Segment per action so BACKTRACK follows its actual visited route.
            prev=tuple(initial)
            for e in t['entries']:
                now=tuple(e['receipt']['cursor_after'])
                x1=px+(prev[0]-X0+.5)*cells;y1=py+(prev[1]-Y0+.5)*cells
                x2=px+(now[0]-X0+.5)*cells;y2=py+(now[1]-Y0+.5)*cells
                ss.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{PALETTE[e["phase"]]}" stroke-width="2.6" opacity=".86" stroke-linecap="round"/>')
                prev=now
            sx=px+(48-X0+.5)*cells;sy=py+(48-Y0+.5)*cells
            ss.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="4" fill="#fff" stroke="#081222" stroke-width="1.3"/>')
            for k,goal in enumerate(((67,48),(46,64))):
                gx=px+(goal[0]-X0+.5)*cells;gy=py+(goal[1]-Y0+.5)*cells
                ss.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="5.5" fill="none" stroke="#fef08a" stroke-width="2"/>')
                ss.append(f'<text x="{gx+7:.1f}" y="{gy-4:.1f}" fill="#fef08a" font-family="monospace" font-size="12">G{k}</text>')
            ex=px+(prev[0]-X0+.5)*cells;ey=py+(prev[1]-Y0+.5)*cells
            ss.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4.4" fill="#fb7185" stroke="#fff" stroke-width="1"/>')
            ss.append(f'<text x="{ox+12}" y="{oy+400}" fill="#b5c6d8" font-family="monospace" font-size="12">64 logical-move gate · {esc(r["reason"])} · {r["switches_seen"]}/2 target switches</text>')
    ss.append('<text x="26" y="1818" fill="#9fb0c8" font-family="monospace" font-size="12">Research overlays are not native terrain. Lines are exact receipt cursor transitions, not inferred/smoothed worm paths.</text>')
    ss.append('</svg>')
    out.write_text('\n'.join(ss)+'\n')
    return candidates

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();print([(layout,key) for layout,key,_ in render(args.run,args.out)])
