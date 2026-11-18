"""Networkless regression gates for frozen Q4 GREYHOUND campaign."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.driv3r.frankenlayers_lab import q4_greyhound as q


class Q4GreyhoundTests(unittest.TestCase):
    def test_strictly_nested_lineage_and_frozen_existing_modes(self):
        self.assertTrue(q.validate_lineage())
        self.assertEqual(tuple(q.PROGRESSION), (
            "greyhound", "grey10_cut", "grey8_cut", "grey6_cut",
            "aero4_no2", "grey3_cut", "grey2_cut"))
        self.assertEqual(list(map(len, q.PROGRESSION.values())), [13,10,8,6,4,3,2])
        self.assertEqual(q.PROGRESSION["grey2_cut"], (0,10))
        self.assertEqual(q.TIERS["upper"], ("grey10_cut","grey8_cut"))
        self.assertEqual(q.TIERS["middle"], ("grey6_cut","aero4_no2"))
        self.assertEqual(q.TIERS["extreme"], ("grey3_cut","grey2_cut"))

    def test_explicit_request_is_bounded_and_source_frozen(self):
        request = json.loads((Path(__file__).with_name("q4-greyhound-request.json")).read_text())
        self.assertEqual(q.validate_request(request), request)
        for key, value in (("max_parallel",2), ("cpu_only",False),
                           ("q4_sha256","BAD"), ("steps_per_episode",300),
                           ("tier","all"), ("native_game",True)):
            mutated = dict(request, **{key:value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                q.validate_request(mutated)
        for key in ("paid_gpu","paid_api","merge","ready","automatic_on_routine_push"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                q.validate_request(dict(request, **{key:True}))
        self.assertEqual(q.validate_request(dict(request, tier="extreme"))["tier"], "extreme")

    def test_historical_blob_identity_refuses_corruption(self):
        self.assertEqual(q.git_blob_sha(b""), "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391")
        self.assertNotEqual(q.git_blob_sha(b"BAD"), q.W0_SHA_BLOB)
        self.assertEqual(q.LEGACY_FILES["experiments/agent_painter/worm/common/worm.py"][1],
                         q.W0_SHA_BLOB)

    def test_source_built_patch_changes_executed_graph_not_weights(self):
        from experiments.driv3r.frankenlayers.patch_runtime import TARGET
        original = ('#include "models.h"\n'
                    '#include "../llama-memory-hybrid-iswa.h"\n\n'
                    'void f() {\n'
                    '    for (int il = 0; il < n_layer; ++il) {\n'
                    '        auto * prev_cur = cur;\n'
                    '        if (il == n_layer - 1 && inp_out_ids) {\n'
                    '            cur = x;\n'
                    '        }\n    }\n}\n')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / TARGET
            path.parent.mkdir(parents=True)
            path.write_text(original)
            r = q.patch_compiled_graph(root, root/"evidence")
            target = path.read_text()
            self.assertTrue(r["decision_graph_only"])
            self.assertFalse(r["base_model_weights_modified"])
            self.assertEqual(r["llama_commit"], q.LLAMA_SHA)
            for name, layers in q.EXTRA_FLOWS.items():
                self.assertEqual(target.count('mode == "'+name+'"'), 1)
                self.assertIn("d1_flow = {"+",".join(map(str,layers))+"}", target)
            self.assertIn("d1_step + 1 == d1_flow.size()", target)
            self.assertIn("d1_has_full", target)
            self.assertIn("d1_has_sliding", target)
            with self.assertRaises(RuntimeError):
                q.patch_compiled_graph(root, root/"evidence")


if __name__ == "__main__":
    unittest.main()
