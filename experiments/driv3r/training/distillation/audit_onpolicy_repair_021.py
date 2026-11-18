"""Independent replay audit for D021 checkpoint receipts and synthetic paintings.

The audit reads saved checkpoint bytes and reconstructs fresh Game instances;
it never consumes archived summary values as simulated evidence.
"""
from __future__ import annotations
import argparse, hashlib, json, tempfile
from pathlib import Path
from .repair import load
from .source_parent_branch_gate import Controller, Gate, parents, sha, js
from .branch_retention_020 import rollout, summary, bootstrap, admissibility

def main(argv=None):
 ap=argparse.ArgumentParser()
 for k in ('d007','d017','q8','d020','out'):ap.add_argument('--'+k,required=True,type=Path)
 a=ap.parse_args(argv);out=a.out
 m=json.loads((out/'d021_metrics.json').read_text())
 for k in ('d007','d017','q8','d020'):
  if sha(getattr(a,k))!=m['source_sha256'][k]:raise AssertionError('source mismatch '+k)
 base=load(a.d007)
 teacher=Controller(base,parents(a.q8),Gate.read(a.d017),.5,.58)
 policies={'d017':teacher,'d020':load(a.d020),
  'd021_mild':load(out/'d021_mild.npz'),'d021_strong':load(out/'d021_strong.npz')}
 stats={};png_verified=0;action_count=0
 with tempfile.TemporaryDirectory() as tmp:
  for label,rg,filename in (
   ('test',range(63001,63017),'d021_test.jsonl'),
   ('confirmation',range(64001,64017),'d021_confirmation.jsonl'),
   ('posthoc_third',range(65001,65017),'d021_third_untouched.jsonl')):
   first=label=='test'
   rows,actions,_=rollout(rg,policies,
    draw=Path(tmp) if first else None,
    receipt_seeds=[63001,63005,63009,63013] if first else [])
   stored=[json.loads(s) for s in (out/filename).read_text().splitlines()]
   if rows!=stored or summary(rows)!=m[label]:raise AssertionError('rollout mismatch '+label)
   stats[label]={'episodes':len(rows),'decisions':sum(r['decisions'] for r in rows)}
   if sum(r['illegal'] for r in rows):raise AssertionError('illegal replay')
   if first:
    expected=[json.loads(s) for s in (out/'d021_actions.jsonl').read_text().splitlines()]
    if actions!=expected:raise AssertionError('receipt divergence')
    action_count=len(expected)
    for f in sorted((out/'pngs').glob('*.png')):
     recreated=Path(tmp)/f.name
     if not recreated.exists() or sha(f)!=sha(recreated):raise AssertionError('PNG divergence '+f.name)
     png_verified+=1
   if label!='posthoc_third':
    if bootstrap(rows,m['selected_diagnostic'],rg)!=m[label+'_ci']:raise AssertionError('CI divergence')
    if admissibility(summary(rows),m['selected_diagnostic'])!=m[label+'_guard']:\
     raise AssertionError('guard divergence')
  audit={'schema':'driv3r.distill.d021-replay-audit/1',
    'runs':stats,'action_receipts':action_count,'png_byte_exact':png_verified,
    'checkpoints_sha256':{n:sha(out/(n+'.npz')) for n in ('d021_mild','d021_strong')},
    'synthetic_only':True,'source_q8_is_style_prior_only':True,'all_legal':True}
 (out/'d021_audit.json').write_text(js(audit)+'\n')
 print(js(audit))
 return audit
if __name__=='__main__':main()
