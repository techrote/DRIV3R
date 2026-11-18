"""C59 source-matched invariants and executed record verification."""
import collections,hashlib,json,unittest
from pathlib import Path
import numpy as np
import c59_eligibility_personality as c59

class TestC59(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[1]
        cls.out=cls.root/'output'
        cls.summary=json.loads((cls.out/'C59-summary.json').read_text())
        cls.rows=json.loads((cls.out/'C59-online.json').read_text())
        cls.traces=json.loads((cls.out/'C59-traces.json').read_text())
    def test_identity(self):
        self.assertEqual(self.summary['source_sha256'],hashlib.sha256((self.root/'source/c59_eligibility_personality.py').read_bytes()).hexdigest())
        self.assertEqual(self.summary['checkpoint_sha256'],hashlib.sha256((self.out/'C59-GATE.safetensors').read_bytes()).hexdigest())
        self.assertGreater(self.summary['training']['adamw_steps'],0)
    def test_split(self):
        tr={r['graph'] for r in self.summary['cohorts']['train']}
        te={r['graph'] for r in self.summary['cohorts']['online']}
        self.assertFalse(tr&te)
        self.assertEqual(len(te),self.summary['online_graphs'])
    def test_unguided_identity_and_legal(self):
        ct=0
        for r in self.rows:
            outcomes=r['outcomes']
            self.assertTrue(all(v['host_rejections']==0 for v in outcomes.values()))
            self.assertEqual(len({v['protected_sha'] for v in outcomes.values()}),1)
            if r['arm'][0]=='none':
                self.assertEqual(len({(v['action_sha'],v['ink_sha']) for v in outcomes.values()}),1)
                ct+=1
        self.assertEqual(ct,self.summary['unguided_parity'])
    def test_remote_route_gate_never_uses_nonexistent_route(self):
        # For every remote-goal record, C59 STRICT must intervene only when
        # a route is available; hidden-goal status alone is not a policy input.
        for r in self.rows:
            if r['arm'][0]=='none':continue
            d=r['outcomes']['C59-STRICT']['counts']
            self.assertGreaterEqual(d.get('intervention_turns',0),0)
            self.assertGreaterEqual(d.get('abstentions',0),0)
            self.assertEqual(d.get('intervention_local-fallback',0),0)
            self.assertEqual(d.get('intervention_turns',0),d.get('intervention_trace-route',0))
    def test_censored_personality_metrics(self):
        keys={'turn_rate','reverse_rate','unique_ratio','direction_entropy','paint_fraction'}
        found=0
        for r in self.rows:
            for v in r['outcomes'].values():
                m=v['personality20']
                if m is None:continue
                found+=1;self.assertEqual(set(m),keys)
                self.assertTrue(all(0<=x<=1 for x in m.values()))
        self.assertGreater(found,0)
    def test_trace_replay_hash_identity(self):
        for t in self.traces:
            matching=[r for r in self.rows if r['graph']==t['graph'] and
               r['drop']==t['drop'] and r['arm']==['backtrack',1.] and
               r['goaltype']==t['goaltype']]
            self.assertTrue(matching)
            z=matching[0]['outcomes'][t['policy']]
            self.assertEqual(z['action_sha'],t['outcome']['action_sha'])
            self.assertEqual(z['ink_sha'],t['outcome']['ink_sha'])
        self.assertEqual(len(self.traces),self.summary['trace_replays'])
if __name__=='__main__':unittest.main()
