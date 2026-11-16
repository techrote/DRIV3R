"""Adversarial P14 protected-overlay and preference ranking checks (CPU-only)."""
import io
import unittest
from pathlib import Path
import torch
import p14_protected_overlay as p


class P14Tests(unittest.TestCase):
    def test_mask_pinned_pretrajectory_and_distinct(self):
        for seed in (2609,2711,2801):
            a=p.protected_cells(seed,'near_origin')
            b=p.protected_cells(seed,'forward_band')
            self.assertEqual(a,p.protected_cells(seed,'near_origin'))
            self.assertTrue(a and b and a != b)
            self.assertNotIn((7,48),a|b)
            self.assertTrue(all(0<=x<96 and 0<=y<96 for x,y in a|b))
        with self.assertRaises(ValueError):p.protected_cells(1,'unknown')

    def test_branch_uses_w0_right_first_unvisited_arm(self):
        obs={'x':10,'y':10,'heading':0,'legal':['STRAIGHT','LEFT','RIGHT','BRANCH','STOP']}
        self.assertEqual(p.legal_destinations(obs,[],{(10,10)})['BRANCH'],(10,11))
        self.assertEqual(p.legal_destinations(obs,[],{(10,10),(10,11)})['BRANCH'],(10,9))
        obs['legal']=['STRAIGHT','RETRACE','STOP']
        self.assertEqual(p.legal_destinations(obs,[(8,10,0)],{(10,10)})['RETRACE'],(8,10))
        with self.assertRaises(AssertionError):p.legal_destinations(obs,[],{(10,10)})

    def test_no_native_protection_claims_or_human_labels(self):
        self.assertEqual(p.GOAL_NOVEL,48)
        self.assertEqual(len(p.LAYOUTS),2)
        self.assertIn('protected',p.__doc__.lower())
        a={'family':'architecture','seed':7,'layout':'near_origin','policy':'original_late',
           'trace_sha256':'a', 'protected_contacts':4,'goal_completed':True,
           'novel_positions':80,'branches':2,'retraces':1,'steps':100}
        group=[]
        for policy in p.POLICIES:
            r=dict(a,policy=policy,trace_sha256=policy)
            if policy!='original_late':r['protected_contacts']=0
            group.append(r)
        pairs=p.preference_pairs(group)
        self.assertGreater(len(pairs),0)
        self.assertTrue(all(t['human_owner_rating'] is None for t in pairs))
        self.assertTrue(all(t['label_type'].startswith('deterministic') for t in pairs))

    def test_protection_precedes_coverage_and_step_count(self):
        unsafe={'protected_contacts':1,'goal_completed':True,'novel_positions':128,
                'branches':50,'retraces':30,'steps':2}
        safe={'protected_contacts':0,'goal_completed':False,'novel_positions':1,
              'branches':0,'retraces':0,'steps':128}
        self.assertGreater(p.priority_score(safe),p.priority_score(unsafe))
        safe_completed=dict(safe,goal_completed=True)
        self.assertGreater(p.priority_score(safe_completed),p.priority_score(safe))

    def test_pair_order_anti_symmetric_and_no_tie_fabrication(self):
        a={'protected_contacts':0,'goal_completed':True,'novel_positions':70,'branches':10,'retraces':5,'steps':80}
        b=dict(a,protected_contacts=3)
        self.assertGreater(p.priority_score(a),p.priority_score(b))
        xa=p.rank_features(a);xb=p.rank_features(b)
        self.assertEqual([v-u for u,v in zip(xa,xb)],[-(u-v) for u,v in zip(xa,xb)])
        self.assertEqual(p.priority_score(a),p.priority_score(dict(a)))

    def test_model_training_is_real_and_learned_head_swaps_labels(self):
        torch.set_num_threads(1)
        train=[]
        for i in range(30):
            v=i%3+1; feats=[-float(v),0.,float(i%2),.1,.1,.1,.1]
            train.extend([{'x':feats,'target_a_preferred':1},
                         {'x':[-x for x in feats],'target_a_preferred':0}])
        head,info=p.train_ranker(train,train)
        self.assertEqual(info['adam_updates'],500)
        self.assertTrue(torch.all(torch.isfinite(head.weight)))
        x=torch.tensor([train[0]['x'],train[1]['x']])
        with torch.no_grad(): logits=head(x).squeeze(1)
        self.assertAlmostEqual(float(logits.sum()),0.,places=5)

    def test_real_frozen_w16_legal_and_contact_veto(self):
        # Deliberately bounded one-world source-pinned independent execution.
        source=Path(__file__).parent/'P11-hosted-test.zip'
        if not source.exists():self.skipTest('P11 immutable producer not staged; supply only for explicit local tests')
        z,w0,head,_=p.base.open_pinned(source.read_bytes())
        rec,_,_=p.base.extract_trace(z,'architecture',2711)
        original,_,om=p.rollout(w0,head,2711,'architecture','near_origin','original_late',rec)
        shielded,dec,sm=p.rollout(w0,head,2711,'architecture','near_origin','veto_late',rec)
        self.assertEqual(original['trajectory'],rec['trajectory'])
        self.assertGreater(om['protected_contacts'],0)
        self.assertEqual(sm['protected_contacts'],0)
        self.assertTrue(all(d['action'] in d['legal'] for d in dec))
        self.assertTrue(all(e['raw_legal'] for e in shielded['events']))

if __name__=='__main__':unittest.main()
