"""Source-matched gallery from executed path receipts only; no hidden geometry."""
import html,json,math,sys
from pathlib import Path

def render(traces,out):
    width=1160;cw=272;ch=208;gap=14;cols=4;rows=math.ceil(len(traces)/cols)
    height=68+rows*(ch+gap)+30
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
       '<rect width="100%" height="100%" fill="#0d1421"/>',
       '<text x="22" y="32" fill="#eff5fa" font-family="monospace" font-size="20">C63 — executed remote-goal recovery paths</text>',
       '<text x="22" y="52" fill="#9eb0bd" font-family="monospace" font-size="11">Source-matched host actions · never reconstructed/imagined walls · dot=start · ring=goal</text>']
    for i,t in enumerate(traces):
        col=i%cols;row=i//cols;x=22+col*(cw+gap);y=64+row*(ch+gap)
        s.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="8" fill="#172435" stroke="#334457"/>')
        title=f'{t["family"]} / {t["policy"]} / {"GOAL" if t["goal"] else "NO GOAL"}'
        s.append(f'<text x="{x+10}" y="{y+18}" fill="{ "#82deb8" if t["goal"] else "#f4ad8e"}" font-family="monospace" font-size="10">{html.escape(title)}</text>')
        coordinates=[t['start']]+[p['p'] for p in t['path']]
        goal=t['target'];allcoords=coordinates+[goal]
        minx,maxx=min(p[0] for p in allcoords),max(p[0] for p in allcoords)
        miny,maxy=min(p[1] for p in allcoords),max(p[1] for p in allcoords)
        span=max(1.,max(maxx-minx,maxy-miny))
        scale= min( (cw-34)/span,(ch-60)/span )
        def xy(p):return (x+17+(p[0]-minx)*scale,y+36+(p[1]-miny)*scale)
        pts=' '.join(f'{px:.1f},{py:.1f}' for px,py in map(xy,coordinates))
        s.append(f'<polyline points="{pts}" stroke="#63c8f4" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" fill="none" opacity=".9"/>')
        ox,oy=xy(t['start']);gx,gy=xy(goal)
        s.append(f'<circle cx="{ox:.1f}" cy="{oy:.1f}" r="3.3" fill="#f5c064"/>')
        s.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="5.5" stroke="#ef87a6" stroke-width="1.5" fill="none"/>')
        s.append(f'<text x="{x+10}" y="{y+ch-11}" fill="#8eabbf" font-family="monospace" font-size="9">steps={len(t["path"])}  action={t["action_sha"][:12]}  ink={t["ink_sha"][:9]}</text>')
    s.append('</svg>');out.write_text('\n'.join(s)+'\n')

if __name__=='__main__':
    data=json.loads(Path(sys.argv[1]).read_text());render(data,Path(sys.argv[2]))
