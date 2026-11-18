"""Source-only wave 20 graph qualification; not real-model inference."""
import ast
import unittest
from pathlib import Path
from experiments.driv3r.frankenlayers_lab.flows import (
    LAB_FLOWS, WAVE20_FLOWS, validate_flows
)
from experiments.driv3r.frankenlayers_lab.wave20_structural import (
    MODES, CANDIDATES, CONTROLS, STEPS, FOLDS, validation_receipt,
    Q8_SHA256, PINNED_LLAMA
)

class TestWave20(unittest.TestCase):
    def test_six_distinct_whole_execution_masks(self):
        self.assertEqual({len(v) for v in WAVE20_FLOWS.values()}, {4,6,8})
        self.assertEqual(len(WAVE20_FLOWS), 6)
        self.assertEqual(len(set(WAVE20_FLOWS.values())), 6)
        self.assertTrue(all(v[0] == 0 and len(set(v)) == len(v) for v in WAVE20_FLOWS.values()))
        self.assertTrue(all(v == tuple(sorted(v)) and max(v) < 24 for v in WAVE20_FLOWS.values()))
        self.assertFalse(set(WAVE20_FLOWS.values()) & set(
            tuple(v) for k,v in LAB_FLOWS.items() if k not in WAVE20_FLOWS))
        validate_flows()

    def test_frozen_flow_identity_not_malleable(self):
        for name, v in WAVE20_FLOWS.items():
            altered = dict(LAB_FLOWS)
            altered[name] = tuple(i+1 if n==1 else i for n,i in enumerate(v))
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_flows(altered)

    def test_true_cpp_execution_path_in_owned_patcher(self):
        src = (Path(__file__).parent/"patch_lab.py").read_text()
        parent = Path(__file__).parent.parent/"frankenlayers"/"patch_runtime.py"
        parent_src = parent.read_text()
        self.assertIn("d1_flow = {cpp_list(layers)}", src)
        self.assertIn("for (size_t d1_step = 0; d1_step < d1_flow.size(); ++d1_step)", parent_src)
        self.assertIn("const int il = d1_flow[d1_step]", parent_src)
        self.assertIn("d1_step + 1 == d1_flow.size()", parent_src)
        self.assertIn("!d1_has_full || !d1_has_sliding", src)

    def test_manual_bounded_run_and_exact_patch_receipt(self):
        self.assertEqual(len(MODES), len(CANDIDATES)+len(CONTROLS))
        self.assertLessEqual(len(MODES)*len(FOLDS)*STEPS, 500)
        self.assertEqual(len(Q8_SHA256),64)
        receipt={"llama_source_commit":PINNED_LLAMA,"decision_only":True,
                 "weights_modified":False,
                 "lab_flows":{m:list(f) for m,f in LAB_FLOWS.items()}}
        validation_receipt(receipt, LAB_FLOWS)
        receipt["lab_flows"]["curl4_cut"]=[0,4,18,23]
        with self.assertRaises(ValueError):
            validation_receipt(receipt, LAB_FLOWS)

if __name__ == "__main__":
    unittest.main()
