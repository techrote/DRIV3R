"""Cheap adversarial tests for matched Q8/Q4 original-trajectory evidence."""
from __future__ import annotations
from pathlib import Path
import json
import tempfile
import unittest
from experiments.driv3r.frankenlayers_lab import expression_recovery as r


class ExpressionRecoveryTests(unittest.TestCase):
    @staticmethod
    def request():
        return json.loads((Path(__file__).with_name("expression-recovery-request.json")).read_text())

    def test_frozen_model_pins_and_host_folds(self):
        self.assertTrue(r.pin_validation())
        self.assertNotEqual(r.Q8_SHA,r.pins.Q4_SHA)
        self.assertEqual(len(r.SCREEN_FOLDS),3)
        self.assertEqual(r.MODES,("stock","greyhound","branchwitch",
                                  "aero4_no2","grey3_cut"))
        self.assertEqual(r.STYLE_NICHES,("Meanderer","Curler","Brancher","Chaotic Artist"))

    def test_request_refuses_unapproved_work_or_invalid_variants(self):
        a=self.request()
        self.assertEqual(r.request_validation(a),a)
        for k,v in (("steps_per_episode",160),("max_parallel",2),
                    ("q8_sha256","0"*64),("q4_bytes",1),
                    ("decoders",["sampled","greedy"]),
                    ("modes",list(reversed(r.MODES))),("phase","chaos"),
                    ("temperature",2),("uniform_mixture",1),
                    ("folds",[[570,"greebles"]]),("native_game",True),
                    ("automatic_push_on_source",True),
                    ("paid_gpu",True),("paid_api",True),
                    ("owner_ratings_supplied",True),
                    ("ready",True),("merge",True)):
            with self.subTest(k=k):
                with self.assertRaises(ValueError):
                    r.request_validation(dict(a,**{k:v}))
        long=dict(a,phase="long",modes=["aero4_no2","grey3_cut"],
                  folds=[list(x) for x in r.LONG_FOLDS],
                  steps_per_episode=r.LONG_STEPS)
        with self.assertRaises(ValueError):
            r.request_validation(long)
        long["screen_artifact_id"]="14429156993"
        self.assertEqual(r.request_validation(long),long)

    def test_common_random_number_and_sampling_legal(self):
        for seed, family in r.SCREEN_FOLDS:
            for tick in (0,1,7,40):
                p=r.common_uniform(seed,family,"greyhound",tick,"sampled")
                self.assertTrue(0<p<1)
                self.assertEqual(p,r.common_uniform(seed,family,"greyhound",tick,"sampled"))
        legal=("STRAIGHT","LEFT","BRANCH","STOP")
        probs={"STRAIGHT":.001,"LEFT":.998,"BRANCH":.001,"STOP":0.}
        q=r.normalize(probs,legal,"sampled")
        self.assertEqual(set(q),set(legal))
        self.assertAlmostEqual(sum(q.values()),1)
        self.assertTrue(q["STOP"]>0)
        self.assertEqual(r.sample(q,.0001),"STRAIGHT")
        self.assertIn(r.sample(q,.95),legal)
        self.assertEqual(r.sample(r.normalize(probs,legal,"greedy"),.5),"LEFT")
        for bad in ({"STRAIGHT":.5,"LEFT":.49,"BRANCH":.0,"STOP":0.},
                    {"STRAIGHT":float("nan"),"LEFT":0.,"BRANCH":0.,"STOP":1.}):
            with self.assertRaises(ValueError):
                r.normalize(bad,legal,"sampled")

    def test_cycle_detector_catches_square_not_just_two_step_loops(self):
        square=[(0,0),(1,0),(1,1),(0,1),(0,0)]
        trajectory=square+square[1:]+square[1:]+square[1:]
        cycle=r.cycle_report(trajectory)
        self.assertEqual(cycle["period"],4)
        self.assertGreaterEqual(cycle["repetitions"],3)
        straight=[(i,0) for i in range(50)]
        self.assertIsNone(r.cycle_report(straight)["period"])
        toggle=[(0,0),(1,0)]*8
        self.assertEqual(r.cycle_report(toggle)["period"],2)

    def test_original_executed_steps_not_invented(self):
        world={
          "trajectory":[[2,2],[3,2],[3,3],[2,3],[2,2]],
          "events":[],
          "step_budget":4,
          "metrics":{"new_steps":3,"total_revisit_contacts":1,
                     "branches":0,"retraces":0,"terminated_by_stop":False}}
        records=[]
        for i,(before,after,choice) in enumerate(zip(world["trajectory"],world["trajectory"][1:],
                                                   ["STRAIGHT","RIGHT","RIGHT","RIGHT"])):
            obs={"x":before[0],"y":before[1],"legal":[choice,"STOP"],"tick":i}
            world["events"].append({"observation":obs,"raw_choice":choice,
                                    "applied":choice,"raw_legal":True,
                                    "position_after":after})
            records.append({"observation":obs,"decoded_choice":choice,
                            "model_argmax":choice,"host_sampling_override":False,
                            "status":"ok","http_status":200,"latency_seconds":1.0})
        desc=r.describe_path(world,records)
        self.assertEqual(desc["new_cells"],3)
        self.assertEqual(desc["unique_positions"],4)
        self.assertEqual(desc["turn_count"],3)
        self.assertFalse(desc["square_loop_collapse"]) # one loop, not sustained
        svg=r.svg_for_ink(world,"BLIND_1")
        self.assertIn("M2,2 L3,2",svg)
        self.assertNotIn("<path d=\"M2,2 L2,2\"",svg)
        world["events"][2]["applied"]="STOP"
        with self.assertRaises(RuntimeError):
            r.describe_path(world,records)

    def test_retrace_has_no_fake_ink_bridge(self):
        world={"trajectory":[[0,0],[1,0],[0,0],[0,1]],
               "events":[{"applied":"STRAIGHT"},{"applied":"RETRACE"},{"applied":"LEFT"}]}
        svg=r.svg_for_ink(world,"anon")
        self.assertIn("M0,0 L1,0",svg)
        self.assertIn("M0,0 L0,1",svg)
        self.assertNotIn('d="M1,0 L0,0"',svg)

    def test_blind_gallery_separates_reveal_and_art(self):
        world={"trajectory":[[0,0],[1,0],[1,1]],
               "events":[{"applied":"STRAIGHT"},{"applied":"RIGHT"}]}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            sv=root/"test.svg"
            sv.write_text(r.svg_for_ink(world,"ANON1"))
            row={"anonymous_id":"ANON1","quant":"q8","mode":"greyhound",
                 "decoder":"sampled","seed":570,"family":"greebles",
                 "source_relpath":"test.svg","metrics":{"new_cells":2}}
            r.build_blind_gallery([row],root,"screen")
            blind=(root/"blind-a4-01.svg").read_text()
            self.assertIn("ANON1",blind)
            self.assertNotIn("greyhound",blind)
            self.assertNotIn("q8",blind)
            self.assertIn("greyhound",(root/"blind-identity-reveal.json").read_text())
            self.assertIsNone(json.loads((root/"blank-owner-ballot.json").read_text())[0]["overall"])


if __name__=="__main__":
    unittest.main()
