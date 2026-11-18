"""C71 source-matched audit. Run with C71_ARCHIVE_ROOT pointing at owner archive root.

No writes to production/repositories. Full ledger and frozen prerequisites required.
"""
from __future__ import annotations
import base64,hashlib,json,os,sys,unittest,zlib
from pathlib import Path
ROOT=Path(os.environ.get('C71_ARCHIVE_ROOT',Path(__file__).resolve().parent)).resolve()
sys.path.insert(0,str(ROOT/'research'))
import c71_remote_goal_transfer as exp

class C71EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output=ROOT/'results_c71'
        cls.summary=json.loads((cls.output/'C71-summary.json').read_text())
        cls.ledger=json.loads(zlib.decompress(base64.b64decode((cls.output/'C71-ledger.zlib.b64').read_text())))

    def test_exact_source_and_checkpoint_bytes(self):
        s=self.summary
        self.assertEqual(s['source_sha256'],hashlib.sha256((ROOT/'research/c71_remote_goal_transfer.py').read_bytes()).hexdigest())
        self.assertEqual(s['frozen_c70_source_sha256'],hashlib.sha256((ROOT/'research/c70_route_memory_actor.py').read_bytes()).hexdigest())
        for k,h in s['checkpoint_sha256'].items():
            self.assertEqual(h,hashlib.sha256((self.output/(k+'.safetensors')).read_bytes()).hexdigest())

    def test_split_and_exclusion_invariants(self):
        s=self.summary
        train={r['graph'] for r in s['train_graph_ids']}
        test={r['graph'] for r in s['test_graph_ids']}
        self.assertEqual(len(train),10);self.assertEqual(len(test),10)
        self.assertFalse(train & test)
        self.assertEqual({r['family'] for r in s['test_graph_ids']},set(exp.FAMILIES))
        self.assertEqual({r['family'] for r in s['train_graph_ids']},set(exp.FAMILIES))

    def test_unknown_target_mechanism_not_self_report_only(self):
        c70=exp.c70; c70.c60.HORIZON=exp.HORIZON
        base,_,_=c70.c60.c59.c58.c55.c54.c52.c50.c47.c44.prerequisites(ROOT/'frozen/predecessor_C42')
        tr,te,_,_,_=exp.cohort(base)
        self.assertEqual({s[6]['graph_sha'] for s in te},{r['graph'] for r in self.summary['test_graph_ids']})
        for st in tr+te:
            goal=tuple(st[5]);ledger=st[2]
            self.assertNotIn(goal,ledger.visits)
            self.assertNotIn(goal,ledger.neighbors)
            self.assertIsNone(ledger.route(st[0].pos,goal))

    def test_complete_matched_executions(self):
        s=self.summary;rows=self.ledger['rows']
        self.assertEqual(s['conditions'],60)
        self.assertEqual(s['executed_suffixes'],420)
        self.assertEqual(len(rows),60)
        self.assertEqual(len({(r['graph'],r['arm'][1],r['drop']) for r in rows}),60)
        self.assertEqual(sum(z['steps'] for r in rows for z in r['outcomes'].values()),s['host_actions_executed'])
        self.assertEqual(s['host_actions_executed'],13540)
        for r in rows:
            self.assertTrue(r['target_unobserved'] and r['initial_route_unavailable'])
            self.assertEqual(set(r['outcomes']),set(exp.POLICIES))
            self.assertEqual(len({z['protected_sha'] for z in r['outcomes'].values()}),1)
            for z in r['outcomes'].values():
                self.assertEqual(z['host_rejections'],0)

    def test_no_guidance_is_identical_across_policies(self):
        un=[r for r in self.ledger['rows'] if r['arm'][1]==0]
        self.assertEqual(len(un),20)
        for r in un:
            self.assertEqual(len({(z['action_sha'],z['ink_sha']) for z in r['outcomes'].values()}),1)

    def test_training_is_real_and_optimizer_matched(self):
        t=self.summary['training']
        self.assertEqual({x['updates'] for x in t.values()},{36})
        self.assertEqual({len(x['epochs']) for x in t.values()},{3})
        self.assertEqual([e['trajectories'] for e in t['C71-ALIGNED']['epochs']],[40,40,40])
        self.assertNotEqual(t['C71-ALIGNED']['epochs'][1]['actions'],t['C71-SHIFTED']['epochs'][1]['actions'])

    def test_aggregate_derived_from_ledger(self):
        ag=self.summary['aggregate']
        for r in self.ledger['rows']:
            strength=r['arm'][1]
            if not strength:continue
            for p,z in r['outcomes'].items():
                key=f'{r["drop"]}/{strength}/{p}'
                self.assertIn(key,ag)
        for key,vals in ag.items():
            drop,strength,pol=key.split('/')
            r=[r for r in self.ledger['rows'] if str(r['drop'])==drop and str(r['arm'][1])==strength]
            self.assertEqual(vals['goal'],sum(x['outcomes'][pol]['goal'] for x in r))
            self.assertEqual(vals['new_sites'],sum(x['outcomes'][pol]['new_sites'] for x in r))

    def test_memory_and_route_ablation_are_not_oracle_predictions(self):
        rows=[r for r in self.ledger['rows'] if r['arm'][1]==1.]
        self.assertEqual(len(rows),20)
        for r in rows:
            original=r['outcomes']['C71-ALIGNED']; ablated=r['outcomes']['C71-NO-ROUTE']
            self.assertEqual(original['action_sha'],ablated['action_sha'])
            self.assertEqual(original['ink_sha'],ablated['ink_sha'])
            self.assertEqual(ablated['route_matching_actions'],0)

if __name__=='__main__': unittest.main(verbosity=2)
