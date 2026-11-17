# GOAL-RECOVERY-001 — goal-following with bounded recovery observations

**17 November 2026 · DRIV3R Draft/HOLD PR #1 · synthetic DEVELOPMENT only.**

## Question and controlled comparison

Does adding **eight host-observable path/recovery features** to a genuinely trained
spatial-goal policy improve survival and goal control, with exactly matched
parameter count, optimizer updates, training states, teacher labels and legal
menus? `goal_only` receives eight zero-masked recovery features. `goal_recovery`
receives them. Both retain the six explicit spatial-goal features from
GOAL-COND-001. Same initialization seed within each pair, same teacher and
curriculum, same actions offered including `STOP`.

The added features describe remaining recovery capacity, current path depth,
legal empties adjacent to each action's proposed cursor, available free exits
at the current path origin, a recovery-budget/depth comparison, and action-specific
recovery availability or immediate frontier pressure. These are hand-engineered
state signals; the experiment **does not** demonstrate inferred global planning.

TRAIN: eight source-derived ordinary worlds plus four trap-corridor instances,
with two training variants per world; stages 36 → 72 → 128, 60 Adam steps per
stage per checkpoint (180 total). Teacher actions are recorded before injected
bad moves, including forced corridor entry. Both arms receive the same labels,
including explicit return decisions. DEVELOPMENT: four different world seeds
and structural instances, each tested as normal and trap-corridor worlds, two
independent goal variants and horizons 128/160. All DEVELOPMENT actions are
neural, apart from three recorded host-legal forced deviations at nominal
steps 27/68/107. No teacher continuation is used after rollout starts.

No sealed archipelago/staggered_vault heldout family was opened. TRAIN and
DEVELOPMENT share the four source structural *classes*, so this does not prove
unseen-family transfer. `RELOCATE_START` remains an explicitly synthetic
history-return operation, not a native CyberSand teleport or permission to
exceed its 64-logical-move limit.

## RESULTS: 128 fully autonomous DEVELOPMENT trajectories

| Training seed | Controller | Goals reached | Mean unique painted cells | Early STOP | Recovery actions | Mean neural decision latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 615 | Goal only | 91 | 129.844 | 7/32 | 127 | 0.1079 ms |
| 615 | Goal + recovery | **107** | **137.344** | **5/32** | 120 | 0.1296 ms |
| 616 | Goal only | 114 | 130.250 | 10/32 | 46 | 0.1003 ms |
| 616 | Goal + recovery | **118** | **141.062** | **1/32** | 69 | 0.1219 ms |
| **Total** | **Goal only** | **205** | **130.047** | **17/64** | **173** | -- |
| **Total** | **Goal + recovery** | **225** | **139.203** | **6/64** | **189** | -- |

Matched episodes: 64 pairs. Recovery features improved ink in 20 pairs,
matched in 31, and reduced it in 13; goal count improved in 11 pairs, matched
in 53, and decreased in none. All 23 actual STOP events across both arms were
**host-forced** (menu `['STOP']` only): the improvement is from arriving at fewer
states with no available movement/recovery, not a learned STOP-threshold change.
Zero illegal actions or protected/nonempty-cell edits were observed.

The more expensive arm adds ~0.0214 ms per decision on seed 615 and ~0.0216 ms
on seed 616. Results are positive *within this synthetic DEVELOPMENT design*;
only two model seeds were examined, and the supplied features contain explicit
local-feasibility arithmetic. A paired statistical confidence interval or
blind heldout result is not claimed. No d1-2B, LFM2.5-350M or Sprocket-500M
weights were trained; these are 961-parameter tiny NumPy policies.

## Exact evidence, checkpoints and review

| Seed | Controller | Checkpoint SHA-256 |
| --- | --- | --- |
| 615 | Goal only | `d77e27765391052723f7bc960b09a4cbb12e8a5da894f5b03ca3725d8aa12e3a` |
| 615 | Goal + recovery | `b9388d4b96de07f62a786283e8dabde41b0a4750aba7455aa5f0e11dacaa4d0d` |
| 616 | Goal only | `14d793b5536bcba620419d82411baced343dffe815801bf3b27d4955ccc33fa1` |
| 616 | Goal + recovery | `e80797a98d0d205bc304556c1fb40be8f059e4f5e0ff592e42f8f7bc216fc381` |

Training source SHA-256: `goal_recovery_lab.py`
`7a8bf710a644b2a42c7e3fb032036f7aa246665e995683891ce9db5fc3caa290`.
Dependencies: `goal_conditioning_lab.py`
`6e00bfdfbaecc8195a86cb626bbbcbb65184a449adc5b7272d5642553cf02659`;
`world.py` `6801c401c356869ed6535fec5da4a627871cb05fff9c95d6726224a492db678a`;
`policy.py` `f3e6469b592e68c0d3410dca88e107b215f1bdb1c16f7ebd73232ea9470cbb82`;
`train.py` `9ed2c9ac49fa397c12f0b8256e4d4439bd3a9294b002b54276b702a1e3ddcacb`.

Original output includes 4 NPZ checkpoints, full neural action distributions,
host-menu/cursor/material events, 128 original JSON trajectories, source-replayed
PNGs, two SHA256SUMS manifests and per-seed reports. Both independent
checkpoint-restoration and trajectory-reexecution checks passed:
**128/128 episodes**, **17,617/17,617 original decisions**,
**166/166 evidence hashes**, **4/4 restored checkpoints**.
Four focused guardrail tests and explicit probability-tampering rejection
passed. Each campaign took 11.698/11.000 s CPU, peak process RSS
122,888/123,416 KiB. All optimizer steps are real bounded NumPy Adam updates.

Independent critical assessment: the comparison is fair on training labels,
model size and initial states, but uses an engineered recovery feature and a
synthetic teacher whose return choices were specifically made explicit.
Both evaluation and teacher share the synthetic host implementation. The
recorded improvement may depend on that structure; it cannot yet justify
native integration or general-purpose planning claims. Metrics are controlled
DEVELOPMENT, **not structural heldout or native acceptance**.

The source and report live in the owned DRIV3R long-horizon lane. Checkpoints
and binary images remain outside Git history. Local ZIP evidence is available
in the conversation; it is **not** yet an immutable GitHub Actions artifact.
No new workflow, paid API/GPU request, production provider change or external
repository mutation was made.

## Next falsification

Test on a fresh, independently defined *blind* structural heldout cohort,
including variable recovery limits, goals with dead-end bait, and control
for the engineered recovery-budget inequality. Pre-register a target/ink/
STOP composite metric, retain the zero-masked control, and require a native
host-contract comparison before advocating integration.
