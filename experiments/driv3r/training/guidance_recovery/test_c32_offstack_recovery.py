"""Focused C32 tests; synthetic host only, no external checkpoints required."""
import unittest
import torch
import numpy as np
from c32_offstack_recovery import (world,bfs_length,fresh,executed_suffix_probe,
     train,SearchPolicy,BranchHead)
from c28_temporal_option import OptionHead,scores,probs
from signed_feedback import HostAdvice

class FlatScorer(torch.nn.Module):
    def forward(self,x):return torch.zeros(x.shape[:-1],dtype=torch.float32)

class OffStackTests(unittest.TestCase):
    def test_host_executed_geometry_reachable_and_offstack(self):
        for i in range(16):
            for s in ('off-early','off-mid','off-late','shortcut','lure2','lure3','lure4'):
                w=world(i,s,1,100+i)
                self.assertEqual((len(w.history),len(w.events),w.invalid_attempts),(27,27,0))
                self.assertNotIn(w.goal,w.history)
                self.assertIsNotNone(bfs_length(w))
    def test_shortest_paths(self):
        expect={'off-early':30,'off-mid':26,'off-late':23,'shortcut':6,
            'lure2':30,'lure3':30,'lure4':30}
        self.assertEqual({k:bfs_length(world(8,k,1,23)) for k in expect},expect)
    def test_training_labels_are_host_executed(self):
        self.assertEqual(executed_suffix_probe(world(1,'shortcut',-1,9)),1)
        for s in ('lure2','lure3','lure4'):
            self.assertEqual(executed_suffix_probe(world(1,s,-1,9)),0)
    def test_optimizer_determinism(self):
        a,ma,_=train(320011,epochs=2);b,mb,_=train(320011,epochs=2)
        self.assertEqual(ma['features_sha256'],mb['features_sha256'])
        self.assertEqual(ma['losses'],mb['losses'])
        for k,val in a.state_dict().items():self.assertTrue(torch.equal(val,b.state_dict()[k]))
    def test_no_cue_identical_scores(self):
        w=world(8,'lure4',1,77);cue=HostAdvice(w);base=FlatScorer()
        p=SearchPolicy(OptionHead(),BranchHead(),'WATCH')
        menu,x,b,t=p.modify(base,w,cue,.35)
        original=scores(base,w,cue,.35)[1]
        self.assertEqual(p.phase,'idle');self.assertIsNone(t)
        self.assertTrue(np.array_equal(x,original))
    def test_probe_and_retreat_stochastic_caps(self):
        base=FlatScorer()
        for scenario,phase,cap in [('lure4','probe',.88),('off-early','retrace',.985)]:
            w=world(8,scenario,1,77);protected=set(w.protected)
            cue=HostAdvice(w);cue.issue_backtrack(w,1.,ttl=9)
            p=SearchPolicy(OptionHead(),BranchHead(),'RULE')
            menu,logits,_,chosen=p.modify(base,w,cue,1.)
            self.assertEqual(p.phase,phase);self.assertIsNotNone(chosen)
            self.assertLessEqual(float(probs(logits)[chosen]),cap+1e-6)
            self.assertTrue(menu[chosen].legal)
            self.assertEqual(w.protected,protected)
    def test_host_cancel_and_goal_change_release(self):
        base=FlatScorer();w=world(8,'off-early',1,33)
        cue=HostAdvice(w);cue.issue_backtrack(w,1.,ttl=9)
        p=SearchPolicy(OptionHead(),BranchHead(),'WATCH')
        p.modify(base,w,cue,.35);self.assertEqual(p.phase,'retrace')
        cue.cancel(w);p.modify(base,w,cue,.35)
        self.assertEqual(p.phase,'released')
        w2=world(8,'off-early',1,34);cue2=HostAdvice(w2)
        cue2.issue_backtrack(w2,1.,ttl=9)
        p2=SearchPolicy(OptionHead(),BranchHead(),'WATCH');p2.modify(base,w2,cue2,.35)
        w2.goal=(12,12);p2.modify(base,w2,cue2,.35)
        self.assertEqual(p2.phase,'released')

if __name__=='__main__':unittest.main()
