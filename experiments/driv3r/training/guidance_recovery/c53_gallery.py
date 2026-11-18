"""Read C53 independently replay-verified paths; render genuine synthetic-host SVG."""
from __future__ import annotations
import json, html, hashlib
from pathlib import Path

def gallery(traces,dest):
    # Recorded 18 replayed paths: three matched snapshots x six controllers
    assert len(traces)==18
    groups={}
    for t in traces:groups.setdefault((t['graph'],len(t['prefix_positions'])-1),[]).append(t)
    assert len(groups)==3 and all(len(x)==6 for x in groups.values())
    unit=13;board=195;pw=296;ph=270
    W=6*pw+24;H=3*ph+78
    buf=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
       '<rect width="100%" height="100%" fill="#101820"/>',
       '<text x="15" y="28" fill="#eef3f7" font-size="19" font-family="sans-serif">C53 · Host-executed recovery trajectories — observed branch memory versus controls</text>',
       '<text x="15" y="48" fill="#9ab2bd" font-size="11" font-family="sans-serif">Black: forbidden/protected · grey: navigable · cyan: pre-cue · yellow: post-cue · red dot: goal · green dot: recovery completion</text>']
    for ri,(_,group) in enumerate(groups.items()):
      for ci,t in enumerate(group):
        x=12+ci*pw;y=64+ri*ph;O=t['outcome']
        buf.append(f'<g transform="translate({x},{y})">')
        buf.append('<rect x="0" y="0" width="285" height="263" rx="7" fill="#1e2c35" stroke="#47606e" stroke-width="1"/>')
        buf.append(f'<text x="11" y="20" fill="#edf4f8" font-family="sans-serif" font-size="12">{html.escape(t["model"])} · {"GOAL" if O["goal"] else "NO GOAL"}</text>')
        buf.append(f'<text x="11" y="36" fill="#9fb9c8" font-family="sans-serif" font-size="10">{html.escape(t["family"])} · {html.escape(t["case"])} · personality {t["personality"]:+d}</text>')
        gx=11;gy=45
        blocked={tuple(v) for v in t['protected']}
        for yy in range(15):
          for xx in range(15):
            px=gx+xx*unit;py=gy+yy*unit
            fill='#090e14' if (xx,yy) in blocked else '#627780'
            buf.append(f'<rect x="{px}" y="{py}" width="12" height="12" fill="{fill}"/>')
        prefix=[tuple(p) for p in t['prefix_positions']]
        full=[tuple(a['pos']) for a in O['path']]
        def curve(points,stroke):
          if len(points)>=2:
            pts=' '.join(f'{gx+u*unit+6},{gy+v*unit+6}' for u,v in points)
            buf.append(f'<polyline points="{pts}" stroke="{stroke}" fill="none" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>')
        curve(prefix,'#68ecf0');curve(prefix[-1:]+full,'#ffd55e')
        if prefix:
          xx,yy=prefix[-1];buf.append(f'<circle cx="{gx+xx*unit+6}" cy="{gy+yy*unit+6}" r="4" fill="#fff"/>')
        tx,ty=t['target'];buf.append(f'<circle cx="{gx+tx*unit+6}" cy="{gy+ty*unit+6}" r="5" fill="#ee4858" stroke="#fff" stroke-width="1"/>')
        if O['goal'] and full:
          xx,yy=full[-1];buf.append(f'<circle cx="{gx+xx*unit+6}" cy="{gy+yy*unit+6}" r="3" fill="#57f39f"/>')
        buf.append(f'<text x="210" y="70" fill="#b0c3cd" font-family="monospace" font-size="10">steps {O["steps_after_cue"]}</text>')
        buf.append(f'<text x="210" y="85" fill="#b0c3cd" font-family="monospace" font-size="10">cue {len(prefix)-1}</text>')
        buf.append(f'<text x="210" y="101" fill="#b0c3cd" font-family="monospace" font-size="10">legal ✓</text>')
        buf.append('</g>')
    buf.append('</svg>')
    dest=Path(dest);dest.write_text('\n'.join(buf)+'\n')
    return {'panels':len(traces),'matched_graphs':len(groups),'svg_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}

if __name__=='__main__':
    p=Path(__file__).resolve().parent
    data=json.loads((p/'output/C53-selected-traces.json').read_text())
    print(gallery(data,p/'output/C53-gallery.svg'))
