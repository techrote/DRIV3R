# Corpus Factory v19 — adversarial editing and rejection recovery

17 November 2026 · DRIV3R PR #1 Draft/HOLD · DATA ONLY · **synthetic host, not native CyberSand**.

## Scientific question

V18's motif authoring benchmark contains clean oracle rollouts from just eight correlated worlds. V19 asks whether a future real SLM can finish the same protected-cell motif objective **after it inherits off-policy damage**: paint on the wrong branch, wrong materials on both branches, failed traversal into protected cells, or a premature visit to the exit. It must select its authoritative lane from remote ordered history, correct the active lane, erase the inactive lane, and exit. The host and reference oracle are intentionally simple, separate from any trained model.

## Dataset and partitions

| Split | World groups | 12-case families | Episodes | Fully labelled recovery decisions | Intervention modes |
|---|---:|---:|---:|---:|---|
| TRAIN | 6 | 18 | 216 | 10,692 | spot, all, blocked |
| DEV | 4 | 8 | 96 | 4,992 | alternating, blocked |
| HELDOUT | 4 | 12 | 144 | 7,032 | cross, exit, blocked |
| Total | 14 | 38 | 456 | 22,716 | 6 distinct modes |

V19 accepts `createHost(record)` and a consumer can call `v18.observe(host)` / `v18.apply(host, action)` for **every post-intervention** move, edit and erase. The oracle is used only to generate trainer labels and validate reachability. The released `checkpoint_model_input` is the sole permitted initial model input; `base_episode`, mode, privileged host, interventions and oracle labels must not be fed to the model. TRAIN/DEV/HELDOUT have disjoint world IDs and arm-length ranges. Counterfactual families, not individual decisions, are the effective unit of inference.

Each 12-member family shares its latest 64 positions/events, unordered event multiset, current local 7×7 view, legal menu, objective and pre-installed damage, yet has six NORTH-first and six SOUTH-first legal oracle decisions. The 50% restricted-projection ceiling is a **constructed property**, not a learned-model result. The full ordered history determines the branch through v17's finite-state relay grammar; a symbolic finite-state interpreter remains a potential shortcut. V19 **does not** prove complex episodic memory.

The `blocked` intervention simulates two rejected traversals against protected cells; v18's host does not produce a failed-action receipt, so v19 explicitly appends **synthetic** refusal receipts and measures the initial rejection counter. Do not represent these as CyberSand's receipt semantics. All other interventions use v18's legal host-applied movement/brush actions. No native game binaries, fixtures, maps or assets are modified.

## Qualification

- Standalone `node test_recovery_edits_v19.js`: **8/8 sections pass**, covering every intervention, wrong-lane cleanup, protected safety, 12-way aliasing, first-action balance, source dependencies, split-specific heldout modes, model-input separation, SHA-256 manifests, deterministic reproduction and overwrite refusal.
- Two independent full `node recovery_edits_v19.js NEW_DIRECTORY 4` generations produced **byte-identical** six JSONL shards and manifest; full default corpus: 456/456 oracle recoveries complete, 22,716 accepted legal expert decisions, 2,280 expert paint actions, 672 expert erase actions, 1,344 scripted wrong paints, 336 deliberately refused protected moves. No *additional* rejected oracle action, no protected entry and no residual wrong paint in successful terminal states.
- Generator runtime: **3.36 / 3.20 seconds**; peak process RSS **276,652 / 275,960 KiB** on Node 22.16.0 Linux. These are generator measurements, not model latency/RSS.
- Pinned dependency Git blobs: v18 `e120babfa7826a2cfb7deb3866f6b7871ed74a45`, v17 `014a25750f368f68b3397d0e413f7faca5ea4c67`, v13 `b49afca90c307d8126c8f4c1becba667b7ced6f1`.
- Generator blob `bc5158088e2828b423403168b239ee45fcb95d9c`; tests blob `8743251fb978418a2893481a94924a541c6d12b9`; canonical manifest SHA-256 `22aebfac0c83296786cb41d55baa6b15b7ae82b9b0ad247ac956f823eaae9f07`.

## Consumer experiment and limitations

From the corpus directory run:

```sh
node test_recovery_edits_v19.js
node recovery_edits_v19.js /new/empty/output/directory 4
```

For the first model comparison, fit a pinned named SLM on TRAIN only, select on DEV, then freeze settings before a one-time HELDOUT assessment. Let the SLM choose every post-intervention action. Compare full ordered history against last64 and unordered-event controls. Report **world-grouped** paired first action, terminal motif correctness, inactive-lane cleanup, recovery erases, new host refusals, step/path stretch and model inference p50/p95/RSS. Keep checkpoint and tokenizer hashes; do not commit model binaries.

**Failure boundaries:** self-contained synthetic v18 brush/replay host, narrow two-lane corridor grammar, related v17/v18 instruction orders, 14 partially correlated worlds, scripted prelude and adversarial interventions, reference oracle episodes rather than learned behavior, no measured native game compatibility, no actual SLM weight update, no immutable Actions artifact. Exact full-observation deduplication is performed; an independent perceptual near-duplicate audit across stylistic variants is **not** claimed. An especially useful follow-up is a fresh topology/receipt grammar not inherited from v17/v18, which would detect symbolic shortcut overfitting.
