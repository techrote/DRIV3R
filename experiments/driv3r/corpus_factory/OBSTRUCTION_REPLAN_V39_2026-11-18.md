# Corpus Factory v39 — remote dynamic-obstruction replanning

**Scope:** Corpus Factory data only. **Status:** locally replay-qualified, synthetic-not-native, Draft/HOLD PR #1. **No model training, no native CyberSand claims.**

## Research question

Can a policy change its first **legal** movement action when a remote two-edge junction is obstructed, despite an identical present local view, unchanged root legal actions, identical paint contamination, and identical prior host history? v38 had law changes but no dynamic physical obstruction.

## Design and causality

Reuse v32 synthetic Host and its accepted 160-step-plus historical painting prelude. Each world has a second route around the east-arm junction: `(0,1)–(1,1)–(2,1)–(3,1)–(3,0)`. A host-dispatched environment transition then toggles both remote connections `(1,0)–(2,0)` and `(1,0)–(1,1)` from clear to blocked, or explicitly leaves them clear. Starting at `(0,0)`, `MOVE_E` and `MOVE_S` remain both legal under both conditions; local five-cell observations are byte-identical. The public obstruction bulletin, not changed immediate legality, disambiguates the remote route.

For the east-arm wrong-paint recovery, `CLEAR` requires `MOVE_E`, while `BLOCKED` requires `MOVE_S`; reversing the first action has **+2 exact oracle regret in either environment**. Removing the bulletin from the paired model-visible input yields byte-identical inputs but different labels. West-arm recovery supplies an unchanged first-action negative control. The host accepts every subsequent expert action and reaches a clean, successful terminal state. No remote-teleporting agent actions occur; the environment change is an **injected host event**, not an autonomous controller action.

## TRAIN / DEV / HELDOUT (default corpus)

| Partition | World variants | Positive pairs | Negative pairs | Episodes | Policy rows |
|---|---:|---:|---:|---:|---:|
| TRAIN | 32 | 32 | 32 | 128 | 3,488 |
| DEV | 16 | 16 | 16 | 64 | 3,456 |
| HELDOUT | 16 | 16 | 16 | 64 | 4,800 |
| **Total** | **64** | **64** | **64** | **256** | **11,744** |

Unique graph geometries: 16, split-disjoint (8/4/4). The four-way historical wrong-paint location structure is inherited from v32 preparation, but the present v39 paired *contrast* is an obstruction event; v39 does not establish a new four-way memory ceiling. Every record is `{input,action,weight}` only. Zero contradictory complete-input labels; zero cross-split exact-input or geometry overlaps; zero exact TRAIN duplicates at selected scope. Challenge records include teacher-only regret and world pointers for DEV/HELDOUT; never append them to model inputs.

## Reproduction and qualification

```
node obstruction_replan_v39.js /tmp/driv3r_v39_a
node obstruction_replan_v39.js /tmp/driv3r_v39_b
node test_obstruction_replan_v39.js /tmp/driv3r_v39_a /tmp/driv3r_v39_b
```

Two complete local generations produced byte-identical policy/challenge shards and manifests. Independent replay checks confirm 256/256 terminal completions. The audit checks all 11,744 legal labels, counterfactual first-action regret, exact SHA-256, unique input labels, cross-split geometry isolation and teacher field separation. Generation timings 4.32 and 3.90 seconds; peak generator RSS 209,908 and 217,368 KiB. Those are **not** neural-inference numbers.

SHA-256: generator `bf93ff3deb59a255e9cacc28d1203bb51098355082f5e60fd9ee4e31c53a9f36`; test `2194207516431d9f71a3e13a7229dc168ee4f337fb0e706c7f41641d0cccb701`; manifest `a12aabd8f570683ba5b6ff0d5899c570088c0f0ad488eb2ff039fdcadf5d3d27`.

## Limitations and consumer handoff

This is a **small deterministic radial-graph grammar**. All positive pairs use the east-to-south first-action contrast; deterministic rule interpretation can solve it, and environmental timing is scripted. The synthetic junction controls are not part of CyberSand native physics or materials. There is no neural policy, no model weights, no model latency/RSS, no immutable Actions corpus artifact. Do not claim general moving-obstacle skill.

For a *real* SLM, train with TRAIN-only v39, evaluate HELDOUT closed-loop with obstruction bulletin present vs deliberately removed, report terminal repair, route regret, first-action accuracy, illegal preferences, p50/p95 latency, model RSS and exact checkpoint hashes. Compare with v38 material-law adaptation and v36 goal revision to avoid specializing to a single rule cue. Favor more varied loops and dynamic obstruction timings in the next corpus expansion.
