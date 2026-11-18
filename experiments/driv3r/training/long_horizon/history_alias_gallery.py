"""Source-grounded SVG gallery of DEVELOPMENT history-alias pairs."""
from __future__ import annotations
import argparse
from pathlib import Path
from xml.sax.saxutils import escape
from . import history_alias_lab as lab

def draw(path):
    items=list(lab.pairs('development'))[:6]
    W,H=1260,950;out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
      '<rect width="100%" height="100%" fill="#101720"/>',
      '<text x="35" y="40" fill="#edf4fb" font-family="sans-serif" font-size="24" font-weight="bold">DRIV3R · HISTORY-ALIAS-001</text>',
      '<text x="35" y="65" fill="#9bb2ca" font-family="sans-serif" font-size="13">Same ink, terrain, legal menu and cursor · different legal path stack · synthetic DEVELOPMENT</text>']
    for i,pair in enumerate(items):
        x=30+(i%2)*620;y=85+(i//2)*285
        out.append(f'<rect x="{x}" y="{y}" width="600" height="266" rx="8" fill="#1a2634" stroke="#344960"/>')
        sp=pair['spec'];title=f'{sp.family.upper()} / {pair["case"][:11]} · goal {pair["goal"]}'
        out.append(f'<text x="{x+18}" y="{y+27}" fill="#e4edf7" font-family="monospace" font-size="13">{escape(title)}</text>')
        for j,kind in enumerate(('positive','negative')):
            w=pair[kind];ox=x+36+j*290;oy=y+60;size=28
            # Local active stack only, recovered prefix omitted from diagram.
            coords=[(px-48,py-48) for px,py in w.path if px>=48 and py>=48]
            allink=[(px-48,py-48) for px,py in w.ink if 0<=px-48<=3 and 0<=py-48<=3]
            for gx in range(4):
                for gy in range(4):
                    cx=ox+gx*size;cy=oy+gy*size
                    fill='#365c6c' if (gx,gy) in allink else '#263548'
                    out.append(f'<rect x="{cx}" y="{cy}" width="{size-2}" height="{size-2}" fill="{fill}"/>')
            pts=' '.join(f'{ox+vx*size+13},{oy+vy*size+13}' for vx,vy in coords)
            color='#71e7c2' if kind=='positive' else '#ffa978'
            out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
            gx,gy=pair['goal'][0]-48,pair['goal'][1]-48
            out.append(f'<circle cx="{ox+gx*size+13}" cy="{oy+gy*size+13}" r="7" fill="#ffdb73" stroke="#101720" stroke-width="2"/>')
            cx,cy=w.cursor[0]-48,w.cursor[1]-48
            out.append(f'<circle cx="{ox+cx*size+13}" cy="{oy+cy*size+13}" r="6" fill="#fff"/>')
            label=f'{kind.upper()} · depth {pair["recoverable_depth" if kind=="positive" else "unrecoverable_depth"]} · budget 3'
            out.append(f'<text x="{ox}" y="{oy+158}" fill="{color}" font-family="monospace" font-size="13">{label}</text>')
            out.append(f'<text x="{ox}" y="{oy+179}" fill="#c8d3e1" font-family="monospace" font-size="12">{"BACKTRACK" if kind=="positive" else "STOP"} label</text>')
        out.append(f'<text x="{x+18}" y="{y+250}" fill="#8ba3bc" font-family="sans-serif" font-size="11">Yellow = target already inked · white = current cursor · lines = active path history</text>')
    out.append('</svg>')
    Path(path).write_text('\n'.join(out)+'\n')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('output',type=Path);draw(a.parse_args().output)
