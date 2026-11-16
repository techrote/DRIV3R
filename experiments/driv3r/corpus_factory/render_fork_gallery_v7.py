#!/usr/bin/env python3
"""Source-matched PNG gallery of four held-out counterfactual fork pairs.

Optional Pillow dependency. Images display remote protected material, executed
wrong-arm prelude and post-choice *reachable corridor*, not model-driven runs.
"""
import argparse
from collections import deque
from pathlib import Path
from PIL import Image, ImageDraw
from counterfactual_fork_v7 import corridor, route_to_dead_end, HUB, GOAL


def render(destination):
    destination=Path(destination)
    if destination.exists(): raise FileExistsError(destination)
    tile=18;pad=16;header=40
    pw=32*tile+2*pad;ph=24*tile+2*pad+header
    im=Image.new('RGB',(2*pw,4*ph),(248,249,250));d=ImageDraw.Draw(im)
    for row,seed in enumerate(range(30000,30004)):
        for col,good in enumerate(('LEFT','RIGHT')):
            w=corridor(seed,good);bad='RIGHT' if good=='LEFT' else 'LEFT'
            prelude=set(route_to_dead_end(w))
            start=(8,11) if good=='LEFT' else (8,13)
            prev={start:None};todo=deque([start]);goal_path=[]
            while todo:
                x,y=todo.popleft()
                if (x,y)==GOAL:break
                for nxt in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                    if nxt in w['open'] and nxt!=HUB and nxt not in prev:
                        prev[nxt]=(x,y);todo.append(nxt)
            p=GOAL
            while p is not None:
                goal_path.append(p);p=prev[p]
            goal_path=set(goal_path)
            ox=col*pw+pad;oy=row*ph+header+pad
            d.text((ox,oy-header),'HELDOUT '+str(seed)+' / goal via '+good+' / prior dead end '+bad,fill=(10,17,28))
            for y in range(24):
                for x in range(32):
                    cell=(x,y);color=(49,55,64)
                    if cell in w['open']:color=(211,219,224)
                    if cell in prelude:color=(239,137,56)
                    if cell in goal_path:color=(74,190,126)
                    if cell in w['protected']:color=(80,118,212)
                    if cell==HUB:color=(248,220,75)
                    if cell==GOAL:color=(237,86,101)
                    if cell==(5,12):color=(84,157,236)
                    d.rectangle((ox+x*tile,oy+y*tile,ox+(x+1)*tile-1,oy+(y+1)*tile-1),fill=color)
            d.rectangle((ox-1,oy-1,ox+32*tile,oy+24*tile),outline=(160,170,180),width=1)
    destination.parent.mkdir(parents=True,exist_ok=True)
    im.save(destination)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    args=p.parse_args();render(args.out)
