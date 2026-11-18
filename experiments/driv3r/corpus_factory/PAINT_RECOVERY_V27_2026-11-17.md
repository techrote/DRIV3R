# DRIV3R Corpus Factory v27 — delayed wrong-material recovery across a causal gate

17 November 2026. **Synthetic host; not the native CyberSand observation/action ABI; no actual SLM training/inference.** This is Corpus Factory data, on the existing Draft/HOLD DRIV3R PR #1 only.

## Scientific target

V26 generated 256 legal episodes with a live paint-triggered gate and accepted-wrong-branch recovery, but **zero oracle erasures** and only **30/128 clean** episodes where frozen-gate reasoning was harmful. A model could learn branch/retrace without learning to repair contamination. V27 adds matched correct-paint controls and accepted bad-paint trajectories after which the agent must find an off-screen contaminated cell, erase, repaint and finish. Combined wrong-branch-plus-material cases stress compounded errors. This extends the same synthetic v25/v26 topology grammar; it is not a new native-world family.

## Protocol and action legality

The v26 exhaustive reverse-BFS over position, predecessor, 3-cell material state and one-time paint-triggered gate latch supplies all *post-intervention* labels. Each v27 world uses three starts: `correct-paint-control`, `wrong-material-recovery`, and `wrong-branch-and-material-recovery`. The last starts with one **accepted legal but suboptimal MOVE_N/BACKTRACK** followed by one **accepted wrong-material paint**. Controls start with one correct paint. The first accepted paint toggles the publicly surveyed edge, in all three cohorts.

After that first paint, a **scripted 72+ accepted-navigation-action remote displacement** moves the agent at least three graph edges from the paint site. Neither the first paint receipt nor `GATE_TRANSITION` remains in the latest 64 receipts when model-labelled decisions begin. The agent can still read the old receipts in the complete history, and can see a public aggregate painted-correct count. The complete-history restriction has **not** been proven necessary because the aggregate count remains informative and the target-selection grammar is limited. This is a targeted delayed-recovery challenge, not a strict information-theoretic memory guarantee.

The 96-move preparatory history, navigation before first paint and remote delay are **scripted**. Every decision afterwards is host-legal oracle-labelled and replayed to terminal STOP. Wrong paint stays in the material state until the oracle actually uses ERASE. Gate toggling and the incorrect paint are accepted host actions, not injected into a hidden state. The synthetic action/observation protocol is `driv3r.synthetic-causal-gate/26`, not the CyberSand canonical ABI.

## Default corpus

| Split | Groups | Episodes (control / wrong material / compound error) | Labelled decisions | Corrective erasures | Causally consequential episodes |
|---|---:|---:|---:|---:|---:|
| TRAIN | 8 | 64 / 64 / 64 | 3,391 | 128 | 143 |
| DEV | 4 | 32 / 32 / 32 | 2,040 | 64 | 58 |
| HELDOUT | 4 | 32 / 32 / 32 | 2,260 | 64 | 45 |
| **Total** | **16** | **128 / 128 / 128** | **7,691** | **256** | **246** |

**384/384** oracle suffixes reached terminal success with **256 accepted wrong-material prefixes, 256 expert ERASEs and 128 compound wrong-branch prefixes**. Remote starting distance minima are 6 TRAIN, 8 DEV and 11 HELDOUT graph edges; longest labelled suffix was 36 decisions (HELDOUT). This is not model success. Across oracle suffix states, frozen-initial-gate decisions were **939 legally available but strictly suboptimal** and **138 illegal**; these are deterministic counterfactual planner diagnostics, not model errors. The 246 strict-causal episodes are not directly comparable to v26's 30/128 clean figure because v27 starts from altered, remotely displaced states.

## Leakage and weighting

All worlds have partition-disjoint geometry/world SHA-256, and **no complete-input hash appears in multiple splits**. Complete inputs with conflicting action labels: **zero**. TRAIN has **416 exact duplicate** decision inputs, weighted zero on repeat, and another **1,133 near-matched public-state/remote-survey/recent-receipt same-label decisions** downweighted by a conservative semantic key. In total **1,549/3,391 TRAIN records have weight zero**; all 2,040 DEV and 2,260 HELDOUT decisions retain weight one. Contradictory near-aliased actions would be retained rather than thrown away. Exact model inputs contain only host observation, not intervention labels or planner audit fields.

Caveat: repeated `ACTION_ACK.painted_correct` publicly discloses whether the earlier paint was correct, even though the paint location/material and gate event have aged out of the last64 receipts. Consequently, this is **not** a proven 50% local-aliased initial-action challenge. Do not report it as such. The model must still identify which nonlocal cell to repair, but fixed recipe/topology regularities may offer shortcuts.

## Reproduction and files

`node paint_recovery_v27.js NEW_DIRECTORY` produces six newline-delimited episode/decision shards plus `manifest.json`; `--quick` uses one world group per split. Run `node test_paint_recovery_v27.js /path/to/full-corpus` to verify the generated manifest/shard hashes, all stepwise reconstructed SHA-256 inputs, host-legal recommended actions, accepted corrections, terminals and split ownership. `generate()` refuses to overwrite outputs.

- Source SHA-256: `6fe483084103f3d2fcaabf539bd6cf6eea22abf03f16a69dfad4ea2f83343538`.
- Test SHA-256: `d99602ce189539cf8c3f7ffb83e216e89fb4d5e888db415576f6aff9caefd396`.
- Dependency v26 SHA-256: `b64925f57b95a20fc1562748a25ba24d7a6cdc01d11787413544aa26451bdd1c`.
- Dependency v25 SHA-256: `c879024a7c8ac96408a0a507be8f12bc7299b2bc6f8baf6be918e738e5d01a7d`.
- Full generated manifest SHA-256: `2cf3e5ff9d968b1b67a9824f1375146e1923bbb903480f586b10f93e6ad0d08e`.
- Two byte-identical **quick** generations and two independent **full generations** produced byte-identical six JSONL shards and manifest. The complete **384-episode / 7,691-action** validator passed.
- Full generations: **51.92 / 50.80 seconds**, **516,472 / 626,084 KiB peak RSS**; full validation: **35.39 seconds, 515,580 KiB peak RSS**. This is generator/validator CPU usage, **not SLM latency/RSS**.

The six generated shards are local data, not committed checkpoint binaries or immutable Actions artifacts. Recreate from pinned text sources; no automatic training is triggered on GitHub push.

## Limitations and next real-model experiment

No named SLM has been trained or evaluated here, and no native CyberSand actor/material state has been observed. The remote displacement is a repeated synthetic path, the two-workshop loop grammar is narrow, the public count leaks some recent correctness information, and generation still uses scripted prefixes. The frozen-gate baseline is an exact symbolic foil, not a plausible neural-controller performance estimate. Neither LFM nor Sprocket consumer currently has access to pinned root weights and execution in its current restricted environment; do not replace them with another scalar/NumPy surrogate.

A consumer lane should train a **real pinned named SLM adapter** on weighted TRAIN, tune on DEV and evaluate complete HELDOUT trajectories once. Compare correct-paint, wrong-material and compound-error cohorts, strict-causal subsets vs negative controls, full old history vs latest64 vs count-only memory, and frozen-gate versus full symbolic controls. Measure exact cleaned target, remaining wrong paint, terminal quota, false RETRACE/STOP, raw illegal preferences, closed-loop goal completion and p50/p95/p99 inference latency and model peak RSS. Do not promote to CyberSand or claim physical acceptance without owner-run native validation.
