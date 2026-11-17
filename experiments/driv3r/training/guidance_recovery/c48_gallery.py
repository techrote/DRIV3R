"""Render C48 source-derived trajectories; no interpolation or imagined paths."""
import argparse,collections,html,json
from pathlib import Path

def draw(paths, dest, count=5):
    grouped=collections.defaultdict(dict)
    for r in paths:grouped[r['graph']][r['model']]=r
    # Prioritize contrasts in actual goals, then include withheld topology.
    groups=sorted(grouped.values(),key=lambda g:(
        -(max(r['goal_events'] for r in g.values())-min(r['goal_events'] for r in g.values())),
        next(iter(g.values()))['family'],next(iter(g.values()))['graph']))[:count]
    side=248;colgap=14;rowgap=32;top=65
    w=4*(side+colgap)+20;h=len(groups)*(side+rowgap)+top+24
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
         '<rect width="100%" height="100%" fill="#121923"/>',
         '<text x="20" y="26" font-family="sans-serif" font-size="18" fill="#f5f7fa" font-weight="700">C48  ·  replay-verified synthetic worm paths</text>',
         '<text x="20" y="46" font-family="sans-serif" font-size="11" fill="#aab5c3">Each row is one executed world/cue; four policies use the same graph and seed. This is not native CyberSand output.</text>']
    colors={'C34-RESET':'#60b0ff','C40-PAIR':'#f1b661','C48-HARD':'#f67f90','C48-SHUFFLED':'#b89dff'}
    for yi,g in enumerate(groups):
      order=list(colors)
      fam=next(iter(g.values()))['family'];gid=next(iter(g.values()))['graph'][:9]
      for xi,name in enumerate(order):
        rec=g[name];xx=14+xi*(side+colgap);yy=top+yi*(side+rowgap)
        out.append(f'<g transform="translate({xx},{yy})">')
        out.append(f'<rect width="{side}" height="{side}" rx="7" fill="#1b2632" stroke="#344454"/>')
        out.append(f'<text x="9" y="17" font-family="sans-serif" font-size="11" fill="{colors[name]}">{name}</text>')
        title=f'{fam} / {gid} · {rec["goal_events"]} goal(s) · {rec["selected"]}'
        out.append(f'<text x="9" y="31" font-family="sans-serif" font-size="9" fill="#d1dce9">{html.escape(title)}</text>')
        plot=side-24;originx=12;originy=38;cell=plot/15
        out.append(f'<rect x="{originx}" y="{originy}" width="{plot}" height="{plot}" fill="#101823"/>')
        for i in range(1,15):
            p=originx+i*cell
            q=originy+i*cell
            out.append(f'<path d="M{p:.1f} {originy}v{plot} M{originx} {q:.1f}h{plot}" stroke="#354150" stroke-width=".38"/>')
        # Horizontal run-length encoding preserves exact protected pixels with
        # substantially less SVG clutter than one rect for every masked cell.
        protected=set(tuple(q) for q in rec['protected'])
        wall_parts=[]
        for y in range(15):
            x=0
            while x<15:
                if (x,y) not in protected:
                    x+=1;continue
                start=x
                while x<15 and (x,y) in protected:x+=1
                a=originx+start*cell;b=originy+y*cell
                wall_parts.append(f'M{a:.1f},{b:.1f}h{(x-start)*cell:.1f}v{cell:.1f}h{-((x-start)*cell):.1f}z')
        out.append(f'<path d="{" ".join(wall_parts)}" fill="#3c4856"/>')
        points=[rec['initial']]+[p['position'] for p in rec['trajectory']]
        strpts=' '.join(f'{originx+(x+.5)*cell:.2f},{originy+(y+.5)*cell:.2f}' for x,y in points)
        out.append(f'<polyline points="{strpts}" fill="none" stroke="{colors[name]}" stroke-width="2.6" opacity=".9" stroke-linejoin="round" stroke-linecap="round"/>')
        out.append(f'<circle cx="{originx+(points[0][0]+.5)*cell:.2f}" cy="{originy+(points[0][1]+.5)*cell:.2f}" r="3" fill="#f7ffbc"/>')
        out.append(f'<circle cx="{originx+(points[-1][0]+.5)*cell:.2f}" cy="{originy+(points[-1][1]+.5)*cell:.2f}" r="3.5" fill="{colors[name]}" stroke="#fcfcff" stroke-width=".7"/>')
        out.append('</g>')
    out.append('</svg>')
    Path(dest).write_text('\n'.join(out)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('paths');p.add_argument('output');a=p.parse_args()
    draw(json.loads(Path(a.paths).read_text()),a.output)
