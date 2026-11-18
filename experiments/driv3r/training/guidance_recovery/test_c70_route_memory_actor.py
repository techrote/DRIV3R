"""C70 executed-evidence regression audit; local CPU; no paid services."""
from __future__ import annotations
import base64, collections, hashlib, json, os, unittest, zlib
from pathlib import Path

DIR=Path(os.environ.get('C70_OUTPUT',str(Path(__file__).resolve().parent/'out')))
SOURCE=Path(__file__).with_name('c70_route_memory_actor.py')

class C70EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary=json.loads((DIR/'C70-summary.json').read_text())
        cls.ledger=json.loads(zlib.decompress(base64.b64decode((DIR/'C70-ledger.zlib.b64').read_bytes())))
        cls.rows=cls.ledger['rows']

    def test_source_hash_and_checkpoint_receipts(self):
        self.assertEqual(self.summary['source_sha256'],hashlib.sha256(SOURCE.read_bytes()).hexdigest())
        for k,digest in self.summary['checkpoint_sha256'].items():
            self.assertEqual(digest,hashlib.sha256((DIR/(k+'.safetensors')).read_bytes()).hexdigest())

    def test_heldout_graphs_and_stratification(self):
        tr=self.summary['train_graphs'];te=self.summary['test_graphs']
        self.assertEqual(len(tr),10);self.assertEqual(len(te),15)
        ids_tr={r['graph'] for r in tr};ids_te={r['graph'] for r in te}
        self.assertEqual(len(ids_tr),10);self.assertEqual(len(ids_te),15)
        self.assertFalse(ids_tr & ids_te)
        for fam in ('tree','braid','prim','wilson','kruskal'):
            self.assertEqual(sum(x['family']==fam for x in tr),2)
            self.assertEqual(sum(x['family']==fam for x in te),3)
        for r in tr+te:
            self.assertTrue(r['route_absent_full_erasure'])
            self.assertGreaterEqual(r['before_edges']-r['after_full_erasure_edges'],3)

    def test_real_optimizer_and_exposure_caveat(self):
        for mode in ('aligned','shifted'):
            x=self.summary['fitness'][mode]
            self.assertEqual(x['updates'],36)
            self.assertEqual(len(x['epochs']),3)
            self.assertTrue(all(y['actions']>0 and y['trajectories']==40 for y in x['epochs']))
        self.assertNotEqual(self.summary['fitness']['aligned']['epochs'][2]['actions'],self.summary['fitness']['shifted']['epochs'][2]['actions'])

    def test_executed_suffixes_integrity_and_parity(self):
        self.assertEqual(len(self.rows),135)
        self.assertEqual(self.summary['executed_suffixes'],810)
        self.assertEqual(sum(o['steps'] for r in self.rows for o in r['outcomes'].values()),self.summary['actions_executed'])
        for r in self.rows:
            self.assertEqual(len(r['outcomes']),6)
            self.assertEqual(len({o['protected_sha'] for o in r['outcomes'].values()}),1)
            self.assertTrue(all(o['host_rejections']==0 for o in r['outcomes'].values()))
            if r['arm'][0]=='none':
                self.assertEqual(len({(o['action_sha'],o['ink_sha']) for o in r['outcomes'].values()}),1)

    def test_aggregate_reconstruction(self):
        for key,agg in self.summary['aggregate'].items():
            drop,strength,policy=key.split('/')
            rr=[r['outcomes'][policy] for r in self.rows if str(r['drop'])==drop and str(r['arm'][1])==strength]
            self.assertEqual(len(rr),15)
            self.assertEqual(sum(z['goal'] for z in rr),agg['goals'])
            self.assertEqual(sum(z['new_sites'] for z in rr),agg['new_sites'])
            self.assertEqual(sum(z['route_matching_actions'] for z in rr),agg['route_matches'])

    def test_memory_intervention_and_replay_gallery(self):
        self.assertEqual(len(self.ledger['gallery']),18)
        self.assertEqual(self.summary['gallery_traces'],18)
        for g in self.ledger['gallery']:
            rr=next(r for r in self.rows if r['graph']==g['graph'] and r['drop']==g['drop'] and r['arm'][1]==1.)
            expected=rr['outcomes'][g['policy']]
            self.assertEqual((g['action_sha'],g['ink_sha']),(expected['action_sha'],expected['ink_sha']))
            self.assertEqual(g['goal'],expected['goal'])
        self.assertIn('<svg',(DIR/'C70-gallery.svg').read_text()[:150])
        for graph in {r['graph'] for r in self.rows}:
            grp={r['drop']:r for r in self.rows if r['graph']==graph and r['arm'][1]==1.}
            self.assertGreater(grp[0.]['outcomes']['STATIC-ROUTE']['route_available_ticks'],0)
            self.assertLessEqual(grp[1.]['outcomes']['STATIC-ROUTE']['route_available_ticks'],grp[0.]['outcomes']['STATIC-ROUTE']['route_available_ticks'])

if __name__=='__main__':unittest.main()
