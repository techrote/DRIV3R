"""Guardrails for autonomous continuation with host-legal moves and sealed holdout."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import autonomous_continuation_lab as a
from .world import specs

class AutonomousContinuationTests(unittest.TestCase):
    def test_partition(self):
        self.assertTrue(all(d in ('W','S') and h in (6,8,10) and b in (2,4) for d,h,b in a.CASES))
        self.assertNotIn('heldout',str(a.CASES))
        self.assertEqual(a.HORIZON,160)

    def test_host_menu_respected_even_with_recovery_exhaustion(self):
        spec=specs('development',1)[0]
        w,_=a.make_forced(spec,6,'W',2,1,3)
        model=a.policy.Ranker(14)
        for _ in range(12):
            if w.stopped:break
            legal=w.menu(3)
            action,prob=a.student_choice(w,3,160,model)
            self.assertIn(action,legal)
            self.assertEqual(set(prob),set(legal))
            self.assertAlmostEqual(sum(prob.values()),1,places=5)
            w.apply(action,3)

    def test_learned_arms_do_not_call_teacher(self):
        spec=specs('development',1)[0]
        models={'imitation':a.policy.Ranker(17),'on_policy':a.policy.Ranker(17),
                'fork':a.CostRanker(17,'nonlinear_geometry')}
        with patch.object(a,'teacher_choice',side_effect=AssertionError('teacher leakage')):
            for arm in ('imitation','on_policy','oracle_fork_neural'):
                a.episode(spec,models,arm,6,'W',2,3,17)

    def test_checkpoint_and_trace_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'study'
            result=a.run(out,610,updates=2,adapt_updates=2,quick=True)
            self.assertEqual(result['episodes'],5)
            self.assertEqual(a.verify(out)['episodes'],5)
            path=next((out/'traces').glob('*.json'))
            row=json.loads(path.read_text())
            row['rows'][-1]['event']['chosen']='MOVE_E' if row['rows'][-1]['event']['chosen']!='MOVE_E' else 'STOP'
            path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):a.verify(out)

if __name__=='__main__':unittest.main()
