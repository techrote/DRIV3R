"""Generate C49 data-faithful SVG from recorded host-executed trajectories."""
import argparse,html,json
from pathlib import Path

def render(rows,dest):
    # Rows are captured host states/actions, not oracle or artistic approximations.
    G=45;cols=3;cw=410;ch=380
    group=sorted(rows,key=lambda z:(z['graph'],z['prefix_t'],('C18-A0','C34-RESET','C40-PAIR').index(z['model'])))
    n=len(group);nrows=(n+cols-1)//cols
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {cols*cw} {nrows*ch}" width="{cols*cw}" height="{nrows*ch}">',
         '<rect width="100%" height="100%" fill="#0d1220"/>',
         '<style>text{font-family:system-ui,Segoe UI,Arial,sans-serif;fill:#d9e2f3} .meta{font-size:12px;fill:#8398b4} .head{font-size:16px;font-weight:650}</style>']
    hues={'C18-A0':'#ffba79','C34-RESET':'#75d6da','C40-PAIR':'#a7a3ff'}
    for ix,p in enumerate(group):
        col=ix%cols;row=ix//cols;bx=col*cw;by=row*ch
        out.append(f'<g transform="translate({bx+8} {by+8})"><rect width="394" height="364" rx="13" fill="#161e2e" stroke="#324258" stroke-width="1"/>')
        title=f"{p['model']} · {p['family']} {p['case']} · cue at step {p['prefix_t']}"
        out.append(f'<text class="head" x="17" y="27">{html.escape(title)}</text>')
        out.append(f'<text class="meta" x="17" y="47">{p["goal_events"]} goals · {len(p["trajectory"])} suffix actions · host legal</text>')
        sx=38;sy=60;pitch=18.0
        # 15 x 15 simulation domain; orientation kept as authored (no 3D angle).
        out.append(f'<g transform="translate({sx} {sy})">')
        out.append(f'<rect width="270" height="270" fill="#101825" rx="4"/>')
        for j in range(16):
            q=j*pitch
            out.append(f'<path d="M {q} 0 V 270 M 0 {q} H 270" fill="none" stroke="#26364a" stroke-width=".55"/>')
        for x,y in p['protected']:
            out.append(f'<rect x="{x*pitch+1:.1f}" y="{y*pitch+1:.1f}" width="16" height="16" fill="#3c4454"/>')
        a=[tuple(q) for q in p['prefix_positions']]
        b=[tuple(q['position']) for q in p['trajectory']]
        def d(points):return ' '.join(('M' if i==0 else 'L')+f'{pitch*x+9:.1f},{pitch*y+9:.1f}' for i,(x,y) in enumerate(points))
        if len(a)>1:out.append(f'<path d="{d(a)}" fill="none" stroke="#60778d" stroke-width="2.5" opacity=".8"/>')
        if b:out.append(f'<path d="{d([a[-1]]+b)}" fill="none" stroke="{hues[p["model"]]}" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"/>')
        x,y=p['goal'];out.append(f'<rect x="{x*pitch+3}" y="{y*pitch+3}" width="12" height="12" fill="#62ef93" transform="rotate(45 {x*pitch+9} {y*pitch+9})"/>')
        x,y=p['start'];out.append(f'<circle cx="{x*pitch+9}" cy="{y*pitch+9}" r="4" stroke="#fff" fill="#f8da68"/>')
        out.append('</g>')
        out.append(f'<text class="meta" x="17" y="352">graph {p["graph"][:14]} · prefix and suffix executed, never planned</text></g>')
    out.append('</svg>')
    dest=Path(dest);dest.write_text('\n'.join(out)+'\n')
    return {'panels':n,'graphs':len({p['graph'] for p in rows}),'viewbox':[0,0,cols*cw,nrows*ch]}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('records');p.add_argument('svg');args=p.parse_args()
    print(render(json.loads(Path(args.records).read_text()),args.svg))
