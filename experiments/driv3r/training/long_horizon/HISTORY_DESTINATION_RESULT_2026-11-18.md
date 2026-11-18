# HISTORY-DESTINATION-001 — synthetic history-aware action-ranker control

**Run date:** 2026-11-18 UTC. **Outcome: bounded learning success, feature hypothesis negative.**

## Question and experimental design

Can a 1,057-parameter CPU-only action ranker learn to take a genuinely legal
`BACKTRACK` toward a goal on its own path, without losing forward navigation?
Does explicitly encoding the destination of BACKTRACK improve such learning?

Using unchanged remote-Git-SHA-matched `world.py`, `policy.py`, `train.py` and
`goal_*` parent sources, build **host-legally executed synthetic** prefixes
of 7–17 moves. At each state, give the ranker either an earlier cursor position
on its own path as a target (history condition) or a free immediate cardinal
neighbor (forward control). TRAIN uses seeds in 1000–1404; DEVELOPMENT uses
3000–3300; no `archipelago` or `staggered_vault` HELDOUT world is opened.
Goal success is **cursor arrival**, including arrival at previously painted
cells: it is NOT new ink or native painting acceptance.

The labels are deliberately constructed from one- to three-cell geometry and
host legality. They are **not** downstream suffix-policy outcomes and cannot
validate general-purpose recovery intelligence. A 4-feature extension encodes
candidate destination progress and whether the goal/destination is on owned
history. `masked` has the same 1,057 parameters, optimizer budget and initialization
but zeroes those features at training and inference. `visible` uses them; `ablated`
uses a visible-trained checkpoint with features zeroed only at inference.
The frozen earlier goal/recovery ranker is a 961-parameter historical baseline,
not an equal-budget new-training comparator.

Two optimizer seeds (619, 620), three stages (36 → 72 → 128), **48 optimizer
updates per stage** and independently frozen DEVELOPMENT geometries. No
intervention is forced after the initial synthetic prefix; every evaluated
action is selected via a full-menu neural softmax and sent to `World.apply()`.
Synthetic `RELOCATE_START` remains in the host menu and is counted rather than
secretly removed, but its historical teleport-like semantics must not be
construed as CyberSand native legality.

## Measured DEVELOPMENT performance

Each horizon contains **32 history** and **32 forward** episodes per controller,
matched across two seeds. Total 768 episodes and 29,934 recorded decisions.

| Horizon | Frozen history arrivals | Masked history arrivals | Visible history arrivals | Ablated history arrivals | Forward arrivals, all policies |
| --- | ---: | ---: | ---: | ---: | ---: |
| 36 | 0/32 | 32/32 | 32/32 | 32/32 | 32/32 each |
| 72 | 0/32 | 32/32 | 32/32 | 32/32 | 32/32 each |
| 128 | 0/32 | 32/32 | 32/32 | 32/32 | 32/32 each |

The new matched trained models correctly selected the initial direction in
all 192 cases across all horizons, whereas the frozen baseline failed at
all 96 ancestor-target cases. Median/mean history-target arrival is 1.75
decisions in both trained arms, including two- and three-step retraces.

**Critically: feature ablation preserved 100% of the achieved visits.**
The specific history-destination feature addition therefore has **no measured
incremental benefit on this easy constructed fixture**. The trained baseline
can already learn the required action from the original feature set and this
condition's directional correlations. Do not attribute these gains to the
new feature representation or claim transfer to structurally novel recoveries.

The 128-step horizon is an evaluation ceiling, **not sustained 128-step
navigation**: only 64/256 evaluation episodes at that ceiling actually
reached 128 steps. There were 13 `RELOCATE_START` executions across all
policies/horizons, all by the older frozen baseline; they are synthetic only.

## Source and evidence qualification

- CPU-only real Adam updates produced four new 1,057-parameter checkpoints,
  with 144 updates each; two earlier frozen checkpoints were restored.
- Five focused unit tests passed, including TRAIN/DEVELOPMENT isolation,
  action-menu equality, actual previous-position destination, checkpoint
  restoration and all recorded prefix actions being host legal.
- Independent verifier **replayed all 768 recorded DEVELOPMENT metrics**
  from restored checkpoints, including 128 full sampled action/probability/
  receipt trajectories. This is not a native simulator execution.
- Bounded campaign wall time: **24.142 s**;
  observed peak process RSS **112532 KiB**.
- Model-only median inference time among tested checkpoints:
  **0.008096–0.008570 ms**.
- Original checkpoints, source, full compact metrics, 128 sampled transcripts,
  hashes and SVG gallery are retained in a local verification archive.
  Checkpoints are **not** in Git history; immutable Actions artifact publication
  has not been achieved and should not be claimed.

## Scientific limitations / decision

This is a **sanity-check learning milestone**, not demonstrated general
long-horizon planning. TRAIN and DEVELOPMENT share the same four known
world-generating families and the same synthetic target construction; every
ancestor goal is already on the worm's own history. The older model was not
trained on this new objective. The apparently dramatic baseline improvement
is largely explained by a simple new supervised task. Goal-switch regret,
unknown exit, branching in novel labyrinths, budget exhaustion and native
host action-contract equivalence remain untested here.

**Next:** deliberately build paired *history-alias* geometries where the same
cursor, heading and visible target require different BACKTRACK choices
according to historical route topology, and add real held-back outcome-suffix
labels before additional adapter training. Keep structural HELDOUT sealed and
do not change CyberSand, Neurounner, or the immutable `d_common` reference.

**Ownership:** `techrote/DRIV3R` Draft/HOLD PR #1 only; research subtree only.
No new workflow, PR, branch, issue, game change or merge.
