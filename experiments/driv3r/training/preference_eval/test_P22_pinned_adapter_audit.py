"""Adversarial tests for real source and LoRA auditor, no fresh model inference."""
from pathlib import Path
import copy
import tempfile
import unittest
import sys
import zipfile
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import P22_pinned_adapter_audit as audit
import p21_real_lfm_bestiary as prior
SOURCE=HERE/'original_sources'
ARCHIVES={
 'stock':SOURCE/'gp350012-stock-source.zip',
 'gp350006':SOURCE/'gp350012-14244806761.zip',
 'gp350010':SOURCE/'gp350012-14245172930.zip',
}
class P22(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.a=prior.load('gp350006',ARCHIVES['gp350006'])
  cls.b=prior.load('gp350010',ARCHIVES['gp350010'])
 def test_exact_historical_weights(self):
  x=audit.weight_audit(ARCHIVES['gp350006'],ARCHIVES['gp350010'])
  self.assertEqual(x['trainable_elements'],172032)
  self.assertEqual(x['sha256_by_checkpoint']['96'],audit.PARENT)
  self.assertEqual(x['sha256_by_checkpoint']['24'],audit.STEP24)
  self.assertTrue(all(z['step24_delta_frobenius']>0 for z in x['target_metrics']))
 def test_original_episodes_and_shared_observations(self):
  rows=audit.prefix_audit(self.a,self.b)
  self.assertEqual(len(rows),12)
  self.assertEqual(sum(not x['same_full_path'] for x in rows),4)
  self.assertEqual(sum(x['common_legal_branch_logits'] for x in rows),1061)
  self.assertEqual(sum(x['child_branch_wins'] for x in rows),3)
  self.assertEqual({r['family'] for r in rows if r['child_branch_ticks']},{'greebles'})
 def test_illegal_branch_margin_never_scored(self):
  d=copy.deepcopy(next(iter(self.a.values()))['decisions'][0])
  d['obs']['legal']=['STRAIGHT','LEFT','RIGHT','STOP']
  self.assertIsNone(audit.legal_margin(d))
 def test_forged_state_rejected(self):
  other=dict(self.b);key=next(k for k in self.b if k[2]=='greebles' and k[0]==24)
  record=copy.deepcopy(other[key]);record['decisions'][0]['obs']['x']+=1;other[key]=record
  with self.assertRaisesRegex(ValueError,'obs mismatch'):audit.prefix_audit(self.a,other)
 def test_forged_decoding_rejected(self):
  other=dict(self.b);key=next(k for k in self.b if k[2]=='architecture' and k[0]==24)
  record=copy.deepcopy(other[key]);record['decisions'][0]['applied']='STOP';other[key]=record
  with self.assertRaisesRegex(ValueError,'decoder|greedy'):audit.prefix_audit(self.a,other)
 def test_nested_parent_tamper_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'bad.zip'
   with zipfile.ZipFile(ARCHIVES['gp350010']) as source,zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as bad:
    for m in source.infolist():
     b=source.read(m.filename)
     if m.filename==audit.NESTED_PATH:
      v=bytearray(b);v[-1]^=1;b=bytes(v)
     bad.writestr(m,b)
   with self.assertRaisesRegex(ValueError,'custody mismatch'):audit.weight_audit(ARCHIVES['gp350006'],path)
 def test_byte_exact_output_and_existing_output_refusal(self):
  with tempfile.TemporaryDirectory() as d:
   a,b=Path(d)/'a',Path(d)/'b'
   self.assertEqual(audit.run(ARCHIVES,a),audit.run(ARCHIVES,b))
   self.assertEqual({x.name:audit.digest(x.read_bytes()) for x in a.iterdir()},
                    {x.name:audit.digest(x.read_bytes()) for x in b.iterdir()})
   self.assertEqual(len(list(a.glob('P22_BLIND_A4_*.svg'))),6)
   with self.assertRaisesRegex(ValueError,'already exists'):audit.run(ARCHIVES,a)
if __name__=='__main__':unittest.main()
