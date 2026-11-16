#!/usr/bin/env python3
"""Executable tests for counterfactual-fork v7 synthetic branch control."""
from pathlib import Path
import tempfile
import unittest
from counterfactual_fork_v7 import (ACTIONS,SEEDS,HUB,GOAL,example,corridor,route_to_dead_end,
   validate_path,branch_reaches_goal,check_partitions,features,train,assess,run)


class CounterfactualForkTests(unittest.TestCase):
    def test_executed_prelude_is_legal_and_long(self):
        for seed in (10000,10003,20023,30093):
            for good in ACTIONS:
                world=corridor(seed,good)
                path=route_to_dead_end(world)
                validate_path(world,path)
                self.assertGreaterEqual(len(path),23)
                self.assertNotIn(HUB, path[4:-1])
                self.assertFalse(world['protected'] & world['open'])

    def test_counterfactual_pair_has_same_local_state(self):
        for seed in (10000,20000,30000):
            a=example(seed,'LEFT','TRAIN')
            b=example(seed,'RIGHT','TRAIN')
            self.assertEqual(a['observation'],b['observation'])
            self.assertNotEqual(a['host_observed_memory'],b['host_observed_memory'])
            self.assertNotEqual(a['prelude_trace_sha256'],b['prelude_trace_sha256'])

    def test_goal_path_is_conditional_on_first_applied_branch(self):
        for seed in (10001,10002,10003):
            for good in ACTIONS:
                w=corridor(seed,good)
                self.assertTrue(branch_reaches_goal(w,good))
                bad=next(x for x in ACTIONS if x!=good)
                self.assertFalse(branch_reaches_goal(w,bad))

    def test_partition_collision_rejected(self):
        rows=[example(s,a,sp) for sp,seeds in SEEDS.items() for s in seeds for a in ACTIONS]
        check_partitions(rows)
        rows[0]['split']='HELDOUT'
        with self.assertRaises(ValueError):check_partitions(rows)

    def test_sham_has_identical_features_for_paired_states(self):
        pair=[example(10010,a,'TRAIN') for a in ACTIONS]
        s=features(pair,'sham')
        self.assertTrue((s[0]==s[1]).all())
        m=features(pair,'memory')
        self.assertFalse((m[0]==m[1]).all())
        self.assertEqual(m.shape,s.shape)

    def test_trained_memory_beats_capacity_matched_sham(self):
        train_rows=[example(s,a,'TRAIN') for s in range(10000,10064) for a in ACTIONS]
        dev_rows=[example(s,a,'DEV') for s in range(20000,20016) for a in ACTIONS]
        m=train(train_rows,'memory');s=train(train_rows,'sham')
        self.assertEqual(assess(dev_rows,m,'memory')['accuracy'],1.0)
        self.assertEqual(assess(dev_rows,s,'sham')['accuracy'],0.5)
        self.assertEqual(assess(dev_rows,m,'swap')['accuracy'],0.0)

    def test_export_repeatability_and_no_heldout_examples(self):
        with tempfile.TemporaryDirectory() as root:
            a=Path(root)/'one';b=Path(root)/'two'
            ar=run(a);br=run(b)
            self.assertEqual((a/'report.json').read_bytes(),(b/'report.json').read_bytes())
            self.assertEqual((a/'train_only.jsonl').read_bytes(),(b/'train_only.jsonl').read_bytes())
            self.assertEqual(ar['HELDOUT'],br['HELDOUT'])
            corpus=(a/'train_only.jsonl').read_text()
            self.assertNotIn('HELDOUT',corpus)
            self.assertNotIn('DEV',corpus)
            self.assertEqual(len(corpus.splitlines()),512)
            with self.assertRaises(FileExistsError):run(a)


if __name__=='__main__':unittest.main()
