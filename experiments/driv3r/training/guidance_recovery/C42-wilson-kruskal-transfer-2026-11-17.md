# C42 — Wilson/Kruskal Transfer of Recovery Action Rankers

**Disposition: HOLD / reject C42-PAIR for promotion.** Synthetic-host science only; no CyberSand native/ABI acceptance. Repository scope: `techrote/DRIV3R` PR #1, `experiments/driv3r/training/guidance_recovery/**`. No external-repository writes.

## Falsifiable question

Does the C40/C41 action-ranking improvement persist on **different maze construction algorithms**, and can a new model trained with *matched* Wilson and Kruskal examples improve online goal completion over a non-neural C34-RESET controller and a label-shuffled head while preserving no-guidance behavior and host legality?

## Design

- New Wilson loop-erased random-walk spanning-tree family and independent randomized Kruskal spanning-tree family, both on the 7x7 odd-cell lattice. Unlike C34/C41, they are neither depth-first search nor Prim. Each tree contains 49 rooms and 48 connecting room edges (97 walkable tiles). Wilson's tree is uniformly sampled; Kruskal's is randomized but is **not** claimed uniform.
- A transparent graph rejection rule selects sufficiently long initial paths for host-executed 24-action prefixes. This **biases the challenge population** toward long-path topologies; it is not general maze sampling. Training, blind and online graph identities are disjoint.
- Training: 72 graphs, 216 states, two candidate-ranking heads (aligned executed-suffix labels and labels shuffled within each host-legal menu), and identical random-initialized control. 1,728 CPU AdamW updates **per trained head**; a second deterministic training run reproduced all parameter tensors. The action-conditional features are the existing 34 host-observable values; shortest paths/graph oracle are used to **construct** episodes and compute offline targets only, not at inference.
- Blind offline: 64 graph identities, 192 decision states, 10-step legally executed suffixes; train/blind/online graph hashes all disjoint.
- Online: 32 additional graph identities over two independent seed cohorts, 1,024 matched policy episodes (8 policies × 4 cue arms × 16 graphs × 2 cohorts). Four arms: unguided, weak BACKTRACK, strong BACKTRACK and strong CONTINUE. Host changes goal at creation and sometimes again at step 50. Horizon 78. All policies use the *same* initial state, host legal menu and action-sampling seed in each matched episode.
- Paired graph-cluster bootstrap: 4,000 descriptive resamples. Not a significance claim or a substitute for broader independent replication.

## Principal observed results

| Policy | Guided goal events / 96 episodes | Wilson | Kruskal | Condition regressions vs C34-RESET | Offline blind mean oracle regret |
|---|---:|---:|---:|---:|---:|
| C18-A0 | 19 | 10 | 9 | 4 | — |
| C34-RESET | 19 | 10 | 9 | 0 | — |
| C40-PAIR | **21** | 10 | **11** | **0** | 0.574479 |
| C41-MIX | 19 | 10 | 9 | 1 | 0.534375 |
| C41-SHUFFLED | 18 | 9 | 9 | 1 | 0.185938 |
| **C42-PAIR** | **18** | **10** | **8** | **3** | **0.144271** |
| C42-SHUFFLED | 18 | 10 | 8 | 1 | 0.153646 |
| C42-RANDOM | 17 | 9 | 8 | 2 | 0.192708 |

C42-PAIR has the lowest blind **offline** regret, but completes **one fewer** online guided goal event than RESET and ties SHUFFLED. C42-PAIR vs RESET: −1 event, descriptive graph-cluster 95% interval [−7, +5]. Against C42-SHUFFLED: 0 events, interval [−6, +6]. Three condition groups regress vs RESET. This is a further **offline-to-online generalization failure**, not evidence of beneficial learned recovery.

C40-PAIR has 21 guided goal events and zero condition regressions in this small transfer sample, but performs poorly on offline suffix-regret (0.574479). That discordance is a warning against using the present 10-step offline oracle alone as the training objective. Its two-event improvement is concentrated in Kruskal cases and is not promotion-quality evidence.

## Integrity, legality and personality observations

- **0 synthetic host rejections**, 0 observed protected-mask mutations, and **32/32 exact no-guidance action + ink hash matches for every policy**. One C41-MIX raw illegal top-1 proposal was blocked by the existing host legal-action mask. Synthetic-mask success does not prove native CyberSand legality.
- **16 independent selected-trajectory replays** match their earlier action hashes, ink hashes, goal results and protected-mask hashes. The HTML gallery is drawn directly from these host-executed trace records, **not** generated reference artwork or native screenshots.
- Preferred/opposed turn-associated ink events differ between policies because the trajectories differ (e.g., C34-RESET 212/193, C42-PAIR 216/184). C42's per-decision signed style projection does **not** establish whole-trajectory personality fidelity. Any promotion requires matched, style-conditioned full-trajectory acceptance, not merely no-guidance hash equivalence.
- The generated challenge population has 15×15-grid, protected-cell and host-legal-menu semantics, but is **not** a readback of CyberSand's authoritative native action/observation interface; do not infer native performance.

## Decision and next test

**Reject C42-PAIR for promotion.** Preserve the *negative* online result and the *positive but misleading* blind offline ranking result. Retain C40-PAIR and C34-RESET as controls, not deployed policies. Stop iterating on the same 10-step suffix target solely to reduce its regret. Prefer an online-outcome-calibrated target under multiple independent topology families and issue-specific no-regression gates. Test more independent graph clusters and whole-trajectory personality before interpreting small wins. Native CyberSand evaluation and immutable GitHub Actions artifact custody remain outstanding.

## Reproduction and custody

Run with the historical DRIV3R Python dependencies, the exact prerequisite `.safetensors` files and the recorded root layout; launch `python c42_wilson_transfer.py <archive-root> <output-dir>`. The code runs bounded CPU-only AdamW; it has **no automatic trigger**, network/API calls or GPU charge. Run `pytest -q test_c42_wilson_transfer.py`; render the gallery with `python c42_gallery.py <archive-root> <output-dir>`. Source and compact evidence live in Git; **checkpoints and full evaluated trajectories are not in Git**. The separately retained SHA-256 archive is a useful custody aid but is **not** an immutable Actions artifact or full repository-only reconstruction. The source snapshot, dependencies and scientific caveats remain material requirements.
