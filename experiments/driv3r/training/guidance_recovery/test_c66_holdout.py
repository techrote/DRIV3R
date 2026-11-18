"""C66 executed-host audit against a frozen C65 owner archive (synthetic only)."""
import base64,collections,hashlib,json,os,unittest,zlib
from pathlib import Path

ROOT=Path(os.environ.get('C66_EVIDENCE','/mnt/data/C66-executed'))
C65=Path(os.environ.get('C65_EVIDENCE','/mnt/data/c65_rebuild/lab'))

class ExecutedC66Audit(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.summary=json.loads((ROOT/'C66-summary.json').read_text())
  cls.ledger=json.loads(zlib.decompress(base64.b64decode((ROOT/'C66-ledger.zlib.b64').read_text())))
 def test_cohorts_and_counts(self):
  s=self.summary
  self.assertEqual(s['graph_count'],24)
  self.assertEqual(len(set(s['graph_ids'])),24)
  self.assertEqual(s['synthetic_host_suffixes'],720)
  self.assertEqual(len(self.ledger['records']),120)
  self.assertEqual(len(self.ledger['samples']),36)
  self.assertGreaterEqual(s['excluded_historical_graph_count'],100)
  self.assertEqual(set(s['graph_ids']) & {r['graph'] for r in json.loads((C65/'C65-summary.json').read_text())['graph_receipts'][0]},set())
 def test_real_checkpoint_identity(self):
  for k,v in self.summary['checkpoints'].items():
   self.assertEqual(hashlib.sha256((C65/(k+'.safetensors')).read_bytes()).hexdigest(),v)
 def test_host_legality_and_unguided_parity(self):
  no=0
  for row in self.ledger['records']:
   vals=list(row['outcomes'].values());self.assertEqual(len(vals),6)
   self.assertTrue(all(a['host_rejections']==0 for a in vals))
   self.assertEqual(len({a['protected_sha'] for a in vals}),1)
   if row['arm'][0]=='none':
    no+=1;self.assertEqual(len({(a['action_sha'],a['ink_sha']) for a in vals}),1)
  self.assertEqual(no,24)
 def test_reexecution(self):
  for t in self.ledger['samples']:
   ref=next(r for r in self.ledger['records'] if r['graph']==t['graph'] and r['drop']==0 and r['arm']==['backtrack',1.])['outcomes'][t['policy']]
   self.assertEqual((ref['action_sha'],ref['ink_sha'],ref['goal']),(t['action_sha'],t['ink_sha'],t['goal']))
 def test_aggregate(self):
  s=self.summary
  for drop in ('0.0','0.5'):
   rows=[r for r in self.ledger['records'] if str(r['drop'])==drop and r['arm'][0]!='none']
   self.assertEqual(len(rows),48)
   for p,stats in s['aggregate'][drop]['policies'].items():
    self.assertEqual(stats['goals'],sum(r['outcomes'][p]['goal'] for r in rows))
    self.assertEqual(stats['new_sites'],sum(r['outcomes'][p]['new_sites'] for r in rows))
 def test_gallery(self):
  p=(ROOT/'C66-gallery.svg').read_text()
  self.assertTrue(p.startswith('<svg'))
  self.assertEqual(p.count('<polyline'),30)

if __name__=='__main__':unittest.main()
