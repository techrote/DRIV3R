"""Render C55 actually executed, hash-replayed host paths; SVG contains no invented strokes."""
from __future__ import annotations
import argparse,html,json
from pathlib import Path

def esc(s):return html.escape(str(s))
def render(traces):
    # Groups of three matched policies; retain only complete triplets.
    groups=[]
    for idx in range(0,len(traces)-2,3):
        group=traces[idx:idx+3]
        if len({(a['graph'],a['tick'],tuple(a['target'])) for a in group})==1:
            groups.append(group)
    groups=groups[:9]
    W=1140; H=94+len(groups)*280
    z=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       '<rect width="100%" height="100%" fill="#101722"/>',
       '<text x="24" y="35" fill="#ffffff" font-size="23" font-family="sans-serif" font-weight="700">C55 · Host-observed waypoint recovery</text>',
       '<text x="24" y="60" fill="#bec9d9" font-size="13" font-family="sans-serif">Frozen trained heads on held-out synthetic graphs · executed and action-hash replayed · each triplet shares world, goal and prefix</text>']
    colors={'C34-RESET':'#f4b162','C55-ROUTE':'#7bd6cc','C55-ALIGNED':'#b5aaff'}
    for row,group in enumerate(groups):
       for col,t in enumerate(group):
          x=24+col*375;y=82+row*280; w=351;h=268
          status='GOAL' if t['outcome']['goal'] else 'MISS'
          c=colors[t['policy']]
          z.extend([f'<rect x="{x}" y="{y}" rx="8" width="{w}" height="{h}" fill="#1a2735" stroke="#334152"/>',
              f'<text x="{x+12}" y="{y+22}" font-family="sans-serif" font-size="15" font-weight="bold" fill="{c}">{esc(t["policy"])}   {status}</text>',
              f'<text x="{x+12}" y="{y+41}" font-family="monospace" font-size="10" fill="#b1c1d0">{esc(t["family"])} · {esc(t["case"])} · seed {t["seed"]} · cue {t["tick"]}</text>'])
          px=x+55;py=y+53;scale=13
          def at(q):return (px+q[0]*scale+scale/2,py+q[1]*scale+scale/2)
          z.append(f'<rect x="{px}" y="{py}" width="195" height="195" fill="#111c29" stroke="#506076"/>')
          for gx,gy in t['protected']:
              z.append(f'<rect x="{px+gx*scale}" y="{py+gy*scale}" width="{scale}" height="{scale}" fill="#344456"/>')
          # C54/C55 runtime prefix is recorded separately from older 24-step fixture.
          pts=t['positions']; pth=[a['p'] for a in t['outcome']['path']]
          for seq,stroke,width,opacity in ((pts,'#e6e9ef',1.6,.60),([pts[-1]]+pth,c,2.4,.95)):
              if len(seq)>1:
                  coords=' '.join(f'{at(q)[0]:.1f},{at(q)[1]:.1f}' for q in seq)
                  z.append(f'<polyline fill="none" stroke="{stroke}" opacity="{opacity}" stroke-width="{width}" stroke-linejoin="round" points="{coords}"/>')
          gx,gy=at(t['target'])
          z.append(f'<path d="M {gx:.1f},{gy-6:.1f} l 6,6 -6,6 -6,-6 z" fill="#f5db70" stroke="#1b2330"/>')
          sx,sy=at(pts[-1]);z.append(f'<circle cx="{sx}" cy="{sy}" r="3.5" fill="#ffffff"/>')
          o=t['outcome']
          z.append(f'<text x="{x+265}" y="{y+94}" font-family="monospace" font-size="11" fill="#d5dfed">steps {o["steps"]:>2}</text>')
          z.append(f'<text x="{x+265}" y="{y+117}" font-family="monospace" font-size="11" fill="#d5dfed">dist {o["final_distance"]:>2}</text>')
          z.append(f'<text x="{x+265}" y="{y+140}" font-family="monospace" font-size="11" fill="#d5dfed">reject {o["host_rejections"]}</text>')
          z.append(f'<text x="{x+12}" y="{y+260}" font-family="monospace" font-size="10" fill="#9fb1c3">graph {t["graph"][:16]} · action {o["action_sha"][:16]}</text>')
    z.append('</svg>')
    return '\n'.join(z)+'\n'

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('trace_json');a.add_argument('svg_path')
    v=a.parse_args();Path(v.svg_path).write_text(render(json.loads(Path(v.trace_json).read_text())))
