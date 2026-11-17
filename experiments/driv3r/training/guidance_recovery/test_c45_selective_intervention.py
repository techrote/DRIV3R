"""Bounded C45 unit checks, synthetic-host only."""
import unittest
import numpy as np
import torch
from c39_action_value import ActionHead
from c45_selective_intervention import GateNet, SelectiveHead, shared_features, train

class C45Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(27)
        self.prior=ActionHead().eval()
        self.features=torch.from_numpy(np.random.default_rng(27).standard_normal((5,34)).astype(np.float32))

    def test_shared_features_only_host_candidate_features(self):
        f=shared_features(self.features,self.prior)
        self.assertEqual(tuple(f.shape),(38,))
        self.assertTrue(bool(torch.isfinite(f).all()))

    def test_closed_gate_reverts_exactly_to_reset_score_delta(self):
        gate=GateNet().eval()
        for p in gate.parameters():p.data.zero_()
        gate.net[-1].bias.data.fill_(-100.)
        head=SelectiveHead(self.prior,gate)
        self.assertTrue(bool(torch.equal(head(self.features),torch.zeros(5))))
        self.assertEqual((head.eligible,head.enabled),(1,0))

    def test_open_gate_uses_frozen_prior(self):
        gate=GateNet().eval()
        for p in gate.parameters():p.data.zero_()
        gate.net[-1].bias.data.fill_(100.)
        head=SelectiveHead(self.prior,gate)
        self.assertTrue(bool(torch.equal(head(self.features),self.prior(self.features))))
        self.assertEqual((head.eligible,head.enabled),(1,1))

    def test_optimizer_repeat_and_shuffled_control(self):
        rng=np.random.default_rng(220)
        rec=[(rng.standard_normal(38).astype('float32'),i%2,float((i%2)*2-1),str(i),'tree') for i in range(24)]
        first,s1=train(rec,seed=421,epochs=2)
        second,s2=train(rec,seed=421,epochs=2)
        self.assertEqual(s1,s2)
        for a,b in zip(first,second):
            for k in a.state_dict():self.assertTrue(torch.equal(a.state_dict()[k],b.state_dict()[k]))
        self.assertFalse(torch.equal(first[0].state_dict()['net.0.weight'],first[1].state_dict()['net.0.weight']))

if __name__=='__main__':unittest.main()
