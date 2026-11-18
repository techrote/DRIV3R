"""Research-only C25 online evidence and legality tests; no native ABI claims."""
import copy, unittest
import numpy as np
import torch
from experiment import World
from signed_feedback import HostAdvice
from c25_online_recovery import (online_feedback,observe36,OnlineResidual,Baseline,
                                  sample_prefix,rollout,groups,qualify)
from c24_history_trust import seed_world
from personality_retention_core import load_c12
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PREREQUISITES=('cue_calibrated','adversarial_adapter','personality_gate',
               'topology_transfer','persistent_feedback','signed_feedback','selective_continue')

class OnlineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.prior=load_c12([ROOT/'prerequisites'/('guidance_recovery_'+s+'.safetensors') for s in PREREQUISITES])
    def test_observed_features_zero_before_any_move(self):
        w=seed_world(0,'reverse',1,1234,True);a=HostAdvice(w)
        a.issue_backtrack(w,ttl=9)
        self.assertTrue(np.array_equal(online_feedback(w,a),np.zeros(4,np.float32)))
        cs,x=observe36(w,.35,a)
        self.assertEqual(x.shape,(10,36));self.assertTrue(np.array_equal(x[:,32:],np.zeros((10,4))))
    def test_replay_after_actual_host_actions_only(self):
        w=seed_world(0,'reverse',1,1234,True);a=HostAdvice(w)
        a.issue_continue(w,ttl=9)
        base=Baseline(self.prior,True)
        v,b=sample_prefix(base,w,a,.35,3,456)
        self.assertEqual(b.age,len(v.events));self.assertEqual(v.invalid_attempts,0)
        self.assertEqual(online_feedback(v,b).dtype,np.float32)
        self.assertTrue(np.all(np.abs(online_feedback(v,b))<=1.))
    def test_goal_change_invalidates_past_evidence(self):
        w=World(7,'islands',1);a=HostAdvice(w);a.issue_continue(w)
        w.events.append(('INK',(1,1),(2,1),w.goal));a.age=1
        self.assertNotEqual(float(np.sum(np.abs(online_feedback(w,a)))),0.)
        w.change_goal()
        self.assertTrue(np.array_equal(online_feedback(w,a),np.zeros(4,np.float32)))
    def test_no_cue_and_other_action_scores_identical(self):
        w=seed_world(0,'reverse',1,1357,True);a=HostAdvice(w)
        m=OnlineResidual(self.prior);baseline=Baseline(self.prior,True)
        for cue in ('none','backtrack','continue'):
            v,b=copy.deepcopy((w,a))
            if cue!='none':getattr(b,'issue_'+cue)(v,ttl=9)
            cs,x=observe36(v,.35,b)
            # Inject a nonzero but bounded history signal into a hypothetical menu.
            x[:,32:]=[.6,.2,.3,-.1]
            with torch.no_grad():new=m(torch.from_numpy(x));old=baseline(torch.from_numpy(x))
            self.assertTrue(torch.equal(new[:8],old[:8]));self.assertTrue(torch.equal(new[9:],old[9:]))
            if cue=='none':self.assertTrue(torch.equal(new,old))
            self.assertEqual(len(cs),10)
    def test_architecture_memory_ablation(self):
        a=OnlineResidual(self.prior,True);b=OnlineResidual(self.prior,False)
        x=np.zeros((10,36),np.float32);x[:,2]=0;x[:,4]=1;x[8,2]=1;x[:,26]=1;x[:,27]=1;x[:,29]=1
        x[:,32:]=[.6,.2,.5,-.3]
        with torch.no_grad():
            a.head[-1].bias.fill_(.5);b.head[-1].bias.fill_(.5)
            self.assertEqual(float(a(torch.from_numpy(x))[8]),float(b(torch.from_numpy(x))[8]))
    def test_host_never_writes_protected_material(self):
        p=Baseline(self.prior,True)
        r=rollout(p,0,'bridge',1,.35,'backtrack',1.,23333,20,True)
        self.assertEqual(r['host_rejected'],0);self.assertEqual(r['illegal_raw_top1'],0)
        self.assertEqual(len(r['geometry']['protected'])>0,True)
        self.assertEqual(tuple(r['geometry']['start']),seed_world(0,'bridge',1,23333,False).pos)
    def test_group_completeness(self):
        rows=[]
        for name in ('C12','C18-A0','C25-ONLINE','C25-NOMEM'):
            for outcome in ('reverse','ahead','shortcut','bridge'):
                for cue,strength in (('none',0.),('backtrack',.35),('backtrack',1.),('continue',.35),('continue',1.)):
                    rows.append({'model':name,'index':0,'outcome':outcome,'personality':1,'arm':.35,'directive':cue,'strength':strength,
                    'success':1,'first_goal_progress_step':1,'steps':2,'personality_ink':0,'personality_opportunities':0,
                    'actions_sha256':'equal','ink_sha256':'equal','host_rejected':0,'illegal_raw_top1':0})
        self.assertEqual(len(groups(rows)),20)
        self.assertTrue(all(q['strict_nonregression'] for q in qualify(rows).values()))

if __name__=='__main__':unittest.main()
