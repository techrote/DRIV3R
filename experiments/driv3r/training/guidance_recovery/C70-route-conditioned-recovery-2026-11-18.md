# C70 — host-observed route-conditioned stochastic recovery actor (2026-11-18)

**Status: scientifically negative / HOLD.** This is external synthetic-host model research, **not** native CyberSand testing, not a proposed replacement ABI, and not evidence of production model readiness. Repository write scope is exclusively `experiments/driv3r/training/guidance_recovery/**` in the long-lived DRIV3R Draft/HOLD PR #1.

## Question and causal access

C68 found memory-erasure trials with mostly unchanged actions; C69 then verified that complete route erasure actually destroyed a previously observed route but the frozen C60 neural controller still changed surprisingly rarely. C70 therefore adds an **explicit read of the host-observed route memory** to the decision-time stochastic actor. The only route permitted is `ledger.route(current_position, host_goal)`, assembled from *past host-offered legal action menus*; it never queries the world's true remote graph or hidden shortest path. Every action is sampled from the host's live legal menu and validated by the unchanged host. The historical `d_common` fixture and CyberSand game are not modified.

The six candidate-action features are: next step along a remembered route, novelty, host-published Manhattan goal progress, legal BACKTRACK action, personality-sign-preserving turn and heavily revisited cell. A small six-weight linear actor adds bounded logits to the original stochastic base/controller logits. Strong and weak corrective cues are kept separate; no-guidance adds zero bias. Critically this does **not** make any action mandatory.

## Cohorts, exposure and scientific controls

Ten training graphs and fifteen completely disjoint held-out graphs, balanced 2/3 across tree, braid, prim, Wilson and Kruskal topology. Pre-intervention eligibility required an observed route of at least three edges, six observed menus, 19 executed prefix actions, and a verified destroyed route under complete erasure. Known historical graph hashes including C69 were excluded before selection. The original design seed `700118` had only three eligible prim graphs against a five-graph quota and was abandoned **before suffix execution or training**. The fixed predeclared eligibility conditions were unchanged, and seed `700119` completed the 25-graph quota. This seed change is a selection-method limitation; it was not based on held-out controller results.

Two independently initialized six-weight actor heads underwent genuine CPU AdamW optimizer training on *on-policy, host-executed transitions*: three epochs × twelve updates = **36 updates per head**. Both used the same ten training graphs, forty episode slots per epoch (weak/strong guidance × intact/full erased), identical numerical objective, random generator initialization, and legal action constraints. ALIGNED uses delayed discounted host-derived rewards (novelty, goal-distance progress, reached goal). SHIFTED cyclically rotates reward-to-action credit within each executed episode, as the wrong-credit control. This within-episode shift retains autocorrelation; it is not a perfect no-signal null. On-policy policies diverge after learning, giving different executed action counts (ALIGNED 990/895/826; SHIFTED 990/633/587). **Do not claim matched training-action exposure.** A static route-following bias (no trained head), frozen C60 neural, C60 FRONTIER and C34 RESET provide additional comparators.

Every one of the 15 test graphs was run across no/weak/strong correction, 0/50/100% remembered-menu erasure, and six policies, for **135 condition rows, 810 host-executed policy suffixes and 14,295 actions**. All no-guidance action and paint hashes matched across policy identities; no illegal host actions or protected-cell modifications occurred.

## Held-out goal completion (per 15 graphs)

| Memory erased | Cue strength | RESET | FRONTIER | Frozen C60 | C70 ALIGNED | C70 SHIFTED | STATIC-ROUTE |
|---|---|---:|---:|---:|---:|---:|---:|
| 0% | weak | 8 | 8 | 8 | 11 | 11 | 10 |
| 0% | strong | 7 | 9 | 11 | 11 | 14 | **15** |
| 50% | weak | 8 | 7 | 8 | 11 | 9 | 8 |
| 50% | strong | 7 | 6 | 11 | 8 | 10 | 7 |
| 100% | weak | 8 | 10 | 9 | 11 | 9 | 8 |
| 100% | strong | 7 | 11 | 11 | **7** | 8 | 7 |

The aligned actor improves over RESET in the weak-cue arms, including full-memory erasure, but under strong intact-memory cues it ties frozen C60 at 11/15 and falls behind STATIC-ROUTE (15/15) and the wrong-credit control (14/15). Under strong complete erasure it regresses to 7/15 versus frozen C60's 11/15. None of those results justifies promotion or a generalized memory-recovery claim.

Strong/intact ALIGNED versus STATIC-ROUTE paired difference was **−4 goals** with 0 wins and 4 losses; exploratory 15-graph bootstrap 95% interval [−8,−1] for the sum. Strong/intact ALIGNED versus SHIFTED was **−3** (one win, four losses; descriptive interval [−7,+1]). These intervals are exploratory and unadjusted across many comparisons; an interval excluding zero in one tiny cohort is not a robust guarantee of superiority.

One explanatory feature: trained ALIGNED retains high positive weights for route match (+0.523), novelty (+0.516), legal BACKTRACK (+0.702), and personality-consistent turns (+0.581). SHIFTED has a much stronger novelty feature (+0.888), stronger goal progress (+0.834), and a negative personality-consistent turn coefficient (−0.404). Both remain stochastic; their weights are mechanistically interpretable but do not prove causal feature attribution. STATIC-ROUTE's success on intact observed waypoint targets does not generalize to genuinely unobserved remote goals. Erased-memory changes also alter available route features later in the trajectory.

## Personality and acceptance boundaries

On strong/intact trajectories, ALIGNED has mean turn rate 0.707 and mean painting-action fraction 0.332, against RESET's 0.736 / 0.362; SHIFTED has 0.550 / 0.513. These are crude movement/ink proxies, **not** full Meanderer, Curler, Brancher or Chaotic-artist acceptance. Changing hint strength or memory can change behavior after the first divergent action; goal outcome deltas are paired by graph but not independent trials at each time step.

## Source qualification and custody

- Source: `c70_route_memory_actor.py`, SHA-256 `cac1f5a31b3d1c322a219b19e3987461ff67f4921957fc8325cd4860b2324436`.
- Trained checkpoints: `C70-ALIGNED.safetensors` SHA-256 `d29341a8fa4c11e3f0ca190f0b4f64ab0b70c6629493e42f30fd6e3e44befabf`; `C70-SHIFTED.safetensors` SHA-256 `223e7694b8509459b9856ac00f7cb3493dd91ba9e733d698d984dcdf4fe174ba`.
- Source-matched evidence includes `C70-summary.json`, `C70-ledger.zlib.b64` (complete per-condition outcomes and action/ink hashes across all 135 rows; only 18 selected paths retain complete action-step traces), `C70-gallery.svg` (18 labeled actual host-executed trajectories), and six `test_c70_route_memory_actor.py` regression tests.
- Two fresh local CPU executions reproduced **byte-identical** checkpoint tensors, complete summary, compressed ledger and SVG gallery. The verified source checksum is included within the summary.
- Checkpoint binaries deliberately **not committed to Git**. They are retained in an owner-evaluation ZIP pending immutable GitHub Actions artifact custody (not yet qualified). No automatic training workflow or paid compute was used.

Reproduction requires the frozen predecessor bundle from the C69 owner archive, Python with numpy/torch/safetensors, this source and existing `c69_memory_potency.py`:

```bash
cd /path/to/extracted-research-bundle
C70_ARCHIVE_ROOT="$PWD" python c70_route_memory_actor.py new-run
C70_OUTPUT="$PWD/new-run" python -m unittest test_c70_route_memory_actor -v
```

**Disposition: HOLD, reject promotion.** Next test should stop treating already-observed waypoint routing as evidence of recovery toward unknown spatial goals; isolate weak-guidance improvements on independent larger cohorts, include a memory-feature ablation and wrong-credit matched-action exposure, and qualify native action/observation acceptance before use in Workbench.
