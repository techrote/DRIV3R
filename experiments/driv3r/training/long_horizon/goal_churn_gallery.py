"""Deterministic six-panel original action-receipt SVG, not AI concept artwork."""
from __future__ import annotations
import argparse, json, html
from pathlib import Path
from .goal_churn_lab import ARMS

COLORS=('#b88935','#319d8f','#a46ad4')
CAP=64


def svg(traces):
    W,H=1320,855; items=[]
    items.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    items.append('<rect width="100%" height="100%" fill="#0e1421"/>')
    items.append('<text x="30" y="42" fill="#eef3fa" font-size="27" font-family="sans-serif" font-weight="bold">DRIV3R · two target reversals, one identical World</text>')
    items.append('<text x="30" y="68" fill="#b0bed0" font-size="14" font-family="sans-serif">Executed synthetic host-legal action receipts · DEVELOPMENT pickets 3000 · model seed 711 · goal switches at ticks 8 and 24</text>')
    for i,arm in enumerate(ARMS):
        tr=traces[arm]; row=tr['row']; entries=tr['entries']; col=i%3; rr=i//3
        cx=20+col*438;cy=92+rr*375
        items.append(f'<rect x="{cx}" y="{cy}" width="420" height="355" rx="10" fill="#182437" stroke="#45576d"/>')
        title=arm.replace('_',' ').upper()
        status='GOAL' if row['arrived_final'] else row['reason'].replace('_',' ')
        items.append(f'<text x="{cx+15}" y="{cy+27}" fill="#e8eff9" font-family="sans-serif" font-size="17" font-weight="bold">{html.escape(title)}</text>')
        items.append(f'<text x="{cx+15}" y="{cy+48}" fill="{("#62d4af" if row["arrived_final"] else "#f6a39d")}" font-family="sans-serif" font-size="12">{html.escape(status)} · {row["decisions"]} actions · {row["planner_calls"]} searches</text>')
        # Crop world to a consistent spatial extent, not auto-zoom per controller.
        ox=cx+105; oy=cy+66; scale=5.0; xmin=41;ymin=42
        def xy(p):return (ox+(p[0]-xmin)*scale,oy+(p[1]-ymin)*scale)
        for n in range(49):
            y=oy+n*scale
            items.append(f'<line x1="{ox}" x2="{ox+42*scale}" y1="{y:.2f}" y2="{y:.2f}" stroke="#2b3e52" stroke-width="0.5"/>')
        for n in range(43):
            x=ox+n*scale
            items.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{oy}" y2="{oy+48*scale}" stroke="#2b3e52" stroke-width="0.5"/>')
        path=[(48,48)]
        for e in tr['prelude']:path.append(tuple(e['cursor_after']))
        if len(path)>1:
            pts=' '.join(f'{xy(p)[0]:.1f},{xy(p)[1]:.1f}' for p in path)
            items.append(f'<polyline points="{pts}" fill="none" stroke="#748ca1" stroke-width="2.0" stroke-linecap="round"/>')
        last=path[-1]
        for phase in range(3):
            coords=[last]
            for e in entries:
                if e['phase']==phase:
                    coords.append(tuple(e['receipt']['cursor_after']))
            if len(coords)>1:
                pts=' '.join(f'{xy(p)[0]:.1f},{xy(p)[1]:.1f}' for p in coords)
                items.append(f'<polyline points="{pts}" fill="none" stroke="{COLORS[phase]}" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>')
                last=coords[-1]
        for sw in tr['switches']:
            t=sw['tick']; e=[x for x in entries if x['tick']<t]
            pt=tuple(e[-1]['receipt']['cursor_after']) if e else path[-1]
            x,y=xy(pt); items.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.6" fill="none" stroke="#f5f9fd" stroke-width="1.5"/>')
            items.append(f'<text x="{x+6:.1f}" y="{y-7:.1f}" fill="#fff" font-family="sans-serif" font-size="9">t{t}</text>')
        for label,p,color in (('A',(60,48),'#f4b44e'),('B',(55,56),'#78beec')):
            x,y=xy(p); items.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="#122134"/>')
            items.append(f'<text x="{x+7:.1f}" y="{y+4:.1f}" fill="{color}" font-family="sans-serif" font-size="12" font-weight="bold">{label}</text>')
        end=tuple(entries[-1]['receipt']['cursor_after']) if entries else path[-1]
        x,y=xy(end); items.append(f'<rect x="{x-3:.1f}" y="{y-3:.1f}" width="6" height="6" fill="#ffffff"/>')
        items.append(f'<text x="{cx+15}" y="{cy+336}" fill="#aabbd0" font-family="sans-serif" font-size="11">orange: phase 0 · teal: phase 1 · purple: phase 2 · rings: goal switches</text>')
    items.append('</svg>')
    return '\n'.join(items)+'\n'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); traces={}
    for arm in ARMS:
        f=a.run/'traces'/f'711-pickets-3000-original-0-8-24-{arm}.json'
        traces[arm]=json.loads(f.read_text())
        assert traces[arm]['row']['arm']==arm and traces[arm]['row']['switches_seen']==2
    a.out.write_text(svg(traces))
    print(str(a.out))

if __name__=='__main__':main()
