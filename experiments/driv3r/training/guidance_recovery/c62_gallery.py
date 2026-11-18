"""C62 SVG paths reconstructed from actual host-executed action traces only."""
import collections,html,json,sys
from pathlib import Path
DATA=Path(sys.argv[1])
OUT=Path(sys.argv[2])
traces=json.loads(DATA.read_text())
labels=['C34-RESET','C60-FRONTIER','C62-ALIGNED']
families=['kruskal','wilson','braid']
assert len(traces)==18
parts=['''<svg xmlns="http://www.w3.org/2000/svg" width="1536" height="810" viewBox="0 0 1536 810">
<rect width="1536" height="810" fill="#101821"/>
<text x="28" y="38" font-family="sans-serif" font-size="26" fill="#e6f3f6" font-weight="bold">C62 · Host-executed remote-goal recovery</text>
<text x="29" y="60" font-family="sans-serif" font-size="13" fill="#a8bbca">Paired synthetic-host trajectories · same prefix / cue · circles start, diamonds host-assigned target · no inferred geometry</text>''']
for ri,fam in enumerate(families):
    for di,drop in enumerate([0.,.5]):
        for pi,pol in enumerate(labels):
            matches=[t for t in traces if t['family']==fam and t['drop']==drop and t['policy']==pol]
            assert len(matches)==1
            t=matches[0];idx=di*3+pi
            x0=28+idx*250;y0=86+ri*237
            parts.append(f'<rect x="{x0}" y="{y0}" width="240" height="222" rx="9" fill="#1d2b36" stroke="#39566a"/>')
            parts.append(f'<text x="{x0+12}" y="{y0+24}" font-family="sans-serif" font-size="14" fill="#d7e6ef" font-weight="bold">{html.escape(fam.upper())} / {html.escape(pol)}</text>')
            parts.append(f'<text x="{x0+12}" y="{y0+43}" font-family="sans-serif" font-size="12" fill="#96aebe">memory erased: {int(drop*100)}%  ·  goal: {"YES" if t["goal"] else "no"}</text>')
            coords=[t['start_pos']]+[step['p'] for step in t['path']]
            coords.append(t['target'])
            xs=[p[0] for p in coords];ys=[p[1] for p in coords]
            lo_x,hi_x=min(xs),max(xs);lo_y,hi_y=min(ys),max(ys)
            area=148;dx=max(1,hi_x-lo_x);dy=max(1,hi_y-lo_y);scale=min(area/dx,area/dy)
            fx=lambda x:x0+18+(x-lo_x)*scale+(area-dx*scale)/2
            fy=lambda y:y0+57+(y-lo_y)*scale+(area-dy*scale)/2
            positions=[t['start_pos']]+[step['p'] for step in t['path']]
            points=' '.join(f'{fx(p[0]):.1f},{fy(p[1]):.1f}' for p in positions)
            parts.append(f'<polyline points="{points}" fill="none" stroke="#7cd3c9" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" opacity="0.84"/>')
            sx,sy=map(float,t['start_pos']);tx,ty=map(float,t['target'])
            parts.append(f'<circle cx="{fx(sx):.1f}" cy="{fy(sy):.1f}" r="5" fill="#70b2ff" stroke="#ffffff" stroke-width="1"/>')
            x,y=fx(tx),fy(ty)
            parts.append(f'<path d="M {x:.1f} {y-6:.1f} L {x+6:.1f} {y:.1f} L {x:.1f} {y+6:.1f} L {x-6:.1f} {y:.1f}Z" fill="#ffc16d" stroke="#ffffff" stroke-width="1"/>')
            parts.append(f'<text x="{x0+12}" y="{y0+208}" font-family="monospace" font-size="11" fill="#96aebe">{len(t["path"]):02d} steps · {t["graph"][:12]} · {t["action_sha"][:9]}</text>')
parts.append('<text x="28" y="805" font-family="sans-serif" font-size="11" fill="#8398a9">Source: C62-executed-ledger; 18 deterministic replay-matched trajectories; synthetic-host evidence, NOT native CyberSand acceptance.</text></svg>')
OUT.write_text('\n'.join(parts)+'\n')
print(f'Wrote {OUT} ({len(traces)} actual trajectories)')
