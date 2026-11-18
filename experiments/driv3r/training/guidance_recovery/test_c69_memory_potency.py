"""C69 executed-evidence audits (run after CPU experiment). No GPU needed."""
from __future__ import annotations
import base64, collections, hashlib, json, os, unittest, zlib
from pathlib import Path

OUTPUT=Path(os.environ.get('C69_EXECUTED_DIR','/tmp/C69'))

class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary=json.loads((OUTPUT/'C69-summary.json').read_text())
        cls.ledger=json.loads(zlib.decompress(base64.b64decode((OUTPUT/'C69-ledger.zlib.b64').read_bytes())))
        cls.rows=cls.ledger['rows']

    def test_predeclared_heldout_graphs(self):
        s=self.summary;g=[x['graph'] for x in s['selected_prefixes']]
        self.assertEqual(len(g),20);self.assertEqual(len(set(g)),20)
        self.assertEqual({e['family']:e['selected'] for e in s['eligible']},dict.fromkeys(('tree','braid','prim','wilson','kruskal'),4))
        self.assertTrue(all(x['route_edges']>=3 and x['prefix_menus']>=6 and x['prefix_actions']>=19 for x in s['selected_prefixes']))
        self.assertTrue(all(x['route_absent_full_erasure'] and x['before_edges']-x['after_full_erasure_edges']>=3 for x in s['selected_prefixes']))

    def test_execution_count_and_host_legality(self):
        self.assertEqual(len(self.rows),180);self.assertEqual(self.summary['host_executed_suffixes'],720)
        self.assertEqual(sum(z['steps'] for r in self.rows for z in r['outcomes'].values()),self.summary['host_actions_executed'])
        for r in self.rows:
            self.assertEqual(len(r['outcomes']),4)
            self.assertTrue(all(z['host_rejections']==0 for z in r['outcomes'].values()))
            self.assertEqual(len({z['protected_sha'] for z in r['outcomes'].values()}),1)

    def test_no_guidance_is_identical(self):
        for r in self.rows:
            if r['arm'][0]=='none':
                self.assertEqual(len({(z['action_sha'],z['ink_sha']) for z in r['outcomes'].values()}),1)
        for graph in {r['graph'] for r in self.rows}:
            rr=[r for r in self.rows if r['graph']==graph and r['arm'][0]=='none']
            self.assertEqual(len(rr),3)
            self.assertEqual(len({(r['outcomes']['C60-FROZEN']['action_sha'],r['outcomes']['C60-FROZEN']['ink_sha']) for r in rr}),1)

    def test_causal_memory_manipulation(self):
        for graph in {r['graph'] for r in self.rows}:
            rr={r['drop']:r for r in self.rows if r['graph']==graph and r['arm'][1]==1.}
            a,b=rr[0.],rr[1.]
            self.assertGreater(a['reads']['C60-FRONTIER']['menus_retained'],b['reads']['C60-FRONTIER']['menus_retained'])
            self.assertEqual(a['reads']['C60-FRONTIER']['route_retained'],1)
            self.assertEqual(b['reads']['C60-FRONTIER']['route_retained'],0)
        self.assertEqual(self.summary['paired']['1.0/1.0/C60-FRONTIER']['changed_traces'],10)
        self.assertEqual(self.summary['paired']['1.0/1.0/C60-FROZEN']['changed_traces'],4)

    def test_summary_matches_individual_pairs(self):
        pp=self.ledger['pairs']
        self.assertEqual(len(pp),480)
        for key,a in self.summary['paired'].items():
            drop,arm,policy=key.split('/')
            rows=[p for p in pp if str(p['drop'])==drop and str(p['arm'][1])==arm and p['policy']==policy]
            self.assertEqual(len(rows),a['pairs'])
            self.assertEqual(sum(p['action_changed'] for p in rows),a['changed_traces'])
            self.assertEqual(sum(p['goal_delta'] for p in rows),a['goal_deltas'])

    def test_gallery_provenance(self):
        self.assertEqual(len(self.ledger['gallery']),16)
        self.assertEqual(self.summary['gallery_traces'],16)
        for g in self.ledger['gallery']:
            rr=next(r for r in self.rows if r['graph']==g['graph'] and r['arm'][1]==1. and r['drop']==g['drop'])
            o=rr['outcomes'][g['policy']]
            self.assertEqual((g['action_sha'],g['ink_sha']),(o['action_sha'],o['ink_sha']))
        self.assertIn('<svg',(OUTPUT/'C69-gallery.svg').read_text()[:150])

if __name__=='__main__':unittest.main()
