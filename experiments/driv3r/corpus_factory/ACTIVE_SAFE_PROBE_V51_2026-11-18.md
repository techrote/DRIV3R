# DRIV3R Corpus Factory v51 — active safe calibration after a non-diagnostic surface warning

**18 November 2026 | Data-only lane | Draft/HOLD PR #1 | Synthetic graph/material host ≠ native CyberSand.**

## Scientific question

V50 created genuinely incorrect protected-goal paints when the host changed a material command mapping without warning, followed by erase/recovery. It remained solvable using the most recent physical paint outcome. V51 tests a different, risk-sensitive decision: when the host gives a **non-diagnostic material-surface warning** immediately before painting a mission objective, does the agent independently travel to an already surveyed sacrificial test cell, perform a real physical paint, inspect the outcome, erase the test paint, return, and safely paint the protected objective? The warning reveals neither actual tool mapping nor whether it changed. A rule that blindly trusts the most recent old calibration fails the newly guarded goal in drifted worlds. This still admits a simple symbolic active-probe rule; the value is a held-out data challenge for a real SLM, not proof of learned behavior.

## Action / observation boundaries

* The policy action set is unchanged: `MOVE_N`, `MOVE_S`, `MOVE_E`, `MOVE_W`, `PAINT_A`, `PAINT_B`, `ERASE`, `STOP`; every label must belong to the observed legal menu and be physically accepted by the synthetic host.
* Existing v50 historical mission, survey, dynamic obstacle, material-law calibration and material-outcome receipts are preserved. A new **synthetic** `SURFACE_ALERT` durable observation contains only a surface label, accepted-action timestamp, and previously inspected safe probe coordinate. The teacher has no direct access to the hidden drift flag or actual mapping; it can only infer from real accepted paint outcomes. These alerts are hypothetical research stimuli, not claims about CyberSand's canonical ABI.
* Agent-selectable safe calibration is inside the supervised trajectory, not in a scripted prelude: travel to safe site → `PAINT_A` → observe material from host → `ERASE` → return to current objective → select physically correct paint. Pre-existing historical calibrations and initial survey remain scripted host-accepted operations and are disclosed as limitations.
* The protected objective uses a **synthetic fail-stop**: physically wrong paint following a warning fails the rollout irreversibly instead of silently allowing erase/repaint. This tests whether proactive calibration avoids costly mistakes. **No claim is made that this matches native CyberSand protected-cell mechanics.**
* Two matched versions of an arm show *identical input and action before safe probing*, including historical receipts, but the actual physical outcomes of safe probing differ. At the later target paint, removing the post-warning physical-result receipt makes their observations identical while the correct target paint command differs. This supplies a strict causal discrimination test instead of a mere changed-output heuristic.
* Irrelevant B warnings do not change A painting, and warnings without drift must still trigger safe checking without causing an incorrect target command. An unalerted baseline avoids the extra trip; opposite initial A mappings check the old-calibration shortcut.

## Exact corpus and qualification

| Split | Disjoint worlds | Successful synthetic episodes | Policy decisions | Active safe probe paints | Physically correct goal paints | TRAIN exact repeats loss-masked |
|:--|--:|--:|--:|--:|--:|--:|
| TRAIN | 48 | 432 | 16,816 | 480 | 864 | 7,708 |
| DEV | 16 | 144 | 6,196 | 160 | 288 | 0 |
| HELDOUT | 16 | 144 | 7,358 | 160 | 288 | 0 |
| **Total** | **80** | **720** | **30,370** | **800** | **1,440** | **7,708** |

Each world has nine predefined arms and an immutable evaluator-only causal ledger: A alert/no-change vs A alert/change, B alert/no-change vs B alert/change, both alerts with joint changes, reverse initial mapping contrasts and no-warning/no-change controls. Across all worlds: 80 A physical-outcome pairs, 80 B pairs, 80 joint pairs, 80 reverse-prior pairs, and 160 warning-without-drift negative comparisons. No host-protected goal was damaged in the teacher traces; every safe probe was erased before final completion.

Two independent full generations were byte-identical for all six JSONL shards **and** the manifest; the source/test files are standard-library Node.js with no model dependencies. An independent auditor rebuilt each host episode from the original generator parent (not from the saved teacher trace), recomputed the input and legal-action label using only observation, replayed all 30,370 actions with physical material outcomes, verified every shard SHA-256, and checked every 720 terminal condition. Full generation completed locally on CPU, no paid training. Distinct world geometries from v44–v50 published local release archives were compared by SHA-256 against v51, with **zero overlaps**; same-input conflicting labels and cross-split geometry collisions were rejected.

## Falsifying negative control

A deterministic **frozen-calibration** control uses the previous v50 observation-only teacher, ignoring the new warning. Across 144 HELDOUT episodes it reached 80 successes; it physically damaged the protected target in **all 64 drifted episodes**. The active-probe observable-state reference teacher completed 144/144, with no protected damage and 160 host-executed safe paint probes. These are deterministic symbolic controls, **NOT neural-model accuracies**. Deliberately non-drifted alert arms remain solvable without probes; only the counterfactual drift arms separate active sensing from stale calibration.

## Reproduce and consume

```bash
node active_probe_v51.js /tmp/driv3r-v51-full-NEW
node test_active_probe_v51.js /tmp/driv3r-v51-full-NEW
node benchmark_controls_v51.js
```

The destination must not exist; `--quick` generates two worlds per split for developer preflight. `V51_DEFAULT_MANIFEST.json` pins six shard names, byte lengths, row counts, SHA-256, full generator SHA-256 and all parent source SHA-256. `TRAIN_policy.jsonl` is supervised-only data; feed `input` to a model, use `action` as label, and honour `weight=0` on repeated TRAIN inputs. Never expose `*_challenge_EVALUATOR_ONLY.jsonl` to the training input. Group worlds by source split, seed and geometry SHA across versions. DEV and HELDOUT are unsampled.

## Negative findings and next consumer gate

This remains a **synthetic, discrete, symbolic graph/material game-like host**; its surface warnings, protected-goal fail-stop and action acceptance are not native CyberSand proof. The initial survey and historical calibrations are scripted, while the new safe probe is part of the labelled agent policy. A one-step active-test parser can still solve the challenges; only a genuinely trained root LFM350/Sprocket SLM in autonomous HELDOUT, with memory/probe ablations and measured p50/p95 CPU inference/RSS, would provide actual model evidence. No model weights were changed and no new paid/GPU execution was used. Generated JSONL shards are not yet immutable GitHub Actions artifacts, and ordinary push does not launch training. Checkpoint/model/tokenizer hash preservation belongs to consumer trainers.

Research ownership: only `experiments/driv3r/corpus_factory/**` in `techrote/DRIV3R` shared `research/import-consolidated-worm-labs` branch. PR #1 stays Draft/HOLD, no other-lab write, no merge/Ready/main change, no CyberSand or Neurounner write.
