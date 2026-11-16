# SIDE-EXIT-COST-001 — measured cost-to-recovery without an engineered feasibility bit

Date: 2026-11-16. **DEVELOPMENT only; synthetic, not native. DRAFT/HOLD.**

## Question and intervention

Does a small neural decision model learn a useful recovery-budget-versus-side-exit-distance interaction from *counterfactual host-legal trajectories*, rather than receiving the teacher's explicit `remaining_budget >= return_distance` feature? The preceding RETURN-GEN-001 engineered control achieved perfect initial-fork agreement, but did not establish a playable long-horizon advantage. Here the preferred fork is labelled by the measured score of **both legal branches** under the same fixed 12-decision continuation, not by a hard-coded inequality.

A corridor contains an enclosed lateral side pocket and an explicit spatial goal accessible only through the original painted path. At its terminal cul-de-sac the host menu includes `BACKTRACK`, `RELOCATE_START`, and `STOP`. The worm has to decide whether to spend recovery actions walking back along its ink to the side exit or relocate to start and paint elsewhere. Only the host `World.menu` and `World.apply` authorize actions/material changes. The source is a synthetic copy of the immutable imported DRIV3R `world.py` fixture (host-reference `techrote/cybersand@313b836b...`); **not** evidence of the current native contract, and not an extension of the native 64-move ceiling.

Training uses distinct *train* world seeds/families, E/N orientations, corridor depths 5/7/9, side exits 1/3/5 steps behind the terminal point, and capacities 2/4/6/9. The 36→72→128 curriculum grows the depth/exit-distance set. DEVELOPMENT uses W/S, depths 6/8/10, exits 2/4, capacities 3/7, and 160-decision rollouts. The structurally distinct `heldout` families are not constructed/inspected. Synthetic materials include solid, material 37, protected and own ink. The pocket has side walls and a closed end so a relocation cannot take an external shortcut to its goal.

The two input controls are a **6-parameter linear ranker with budget alone**, and a **6-parameter linear ranker with raw budget, path depth and observed nearest side-exit distance**. The treatment is an **85-parameter tanh MLP** with those same raw inputs. No model sees an explicit feasibility comparison, terminal reward, future trajectories or goal coordinates at its neural fork. Labels come from paired twelve-decision host-legal continuations, whose declared scoring function rewards actually reached goals, additional ink, survival and reduced recovery cost. All three students train from identical examples with 120 CPU Adam updates at each of the three stages. The first stage has only positive labels; this is a significant curriculum imbalance.

**Important limitation:** the trained model chooses one recovery action at the forced junction. The rest of the 160-decision episode is driven by the fixed, imperfect source-derived continuation controller (with three scheduled legal perturbation opportunities). This is **not** a fully neural 160-decision autonomous worm, nor a d1/LFM/Sprocket adapter, nor an independent natural-world generalization test.

## Results — three separate initialized seeds, 160-decision DEVELOPMENT

| Policy | Correct counterfactual fork, combined | Goal reached | Early STOP | Mean painted cells by seed | Mean 12-step regret |
| --- | ---: | ---: | ---: | --- | ---: |
| Budget only, linear | 54/72 | 54/72 | 72/72 | 12.50 / 12.50 / 12.50 | 1.1250 |
| Raw geometry, linear | 54/72 | 54/72 | 72/72 | 12.50 / 12.50 / 12.50 | 1.1250 |
| Raw geometry, 85-param tanh MLP | **72/72** | 54/72 | **56/72** | **50.25 / 47.50 / 50.25** | **0.0000** |

For seeds 600 / 601 / 602 respectively, full 72-episode campaign wall time was **7.883 / 7.848 / 7.772 s**, peak RSS **101,904 / 101,780 / 101,908 KiB**. Per neural fork decision latency was approximately **0.02–0.024 ms linear** and **0.040–0.043 ms MLP**; the full continuation controller is separately timed and excluded from this fork timing. Original trajectories collectively contain **216 episodes and 6,600 re-executed host decisions**. Each seed yielded **three real NumPy checkpoints** and a SHA-256 manifest; 79 files per campaign, 237 verified entries collectively. Every checkpoint was restored, every recorded host action/menu/material transition and fork softmax was replayed, and all three result aggregates were recomputed. Focused plus predecessor guardrails: **14/14 PASS**.

The nonlinear model learned the needed local comparison over the deliberately narrow DEVELOPMENT partition where linear students did not, and its choice improved downstream painted-cell count. However, *all models reached exactly the same number of side goals*, and the best model still stopped early in **56/72** episodes. A budget-aware **analytic threshold** would trivially solve this small family. The score of each fork comes from one seeded 12-step teacher continuation, not an average over stochastic futures or causal native gameplay. The reported gain is not evidence of general-purpose planning, novel spatial reasoning, or reliable 128+ autonomy. All 216 traces reuse one DEVELOPMENT structural family and seed (across geometrical transforms); the three training seeds are not 72 structurally independent worlds.

## Reproduce, inspect and extend

```bash
python -m unittest experiments.driv3r.training.long_horizon.test_side_exit_cost_lab experiments.driv3r.training.long_horizon.test_return_generalization_lab -q
python -m experiments.driv3r.training.long_horizon.side_exit_cost_lab --out /tmp/side-exit-600 --seed 600 --updates 120
python -m experiments.driv3r.training.long_horizon.side_exit_cost_lab --verify /tmp/side-exit-600
```

Use fresh output directories for seeds 601 and 602. Exact hashes for `side_exit_cost_lab.py` and its imported dependencies are recorded in each `report.json`; each `SHA256SUMS.json` hashes every checkpoint and trajectory. Store checkpoint and original trajectory evidence outside Git history in an immutable, opt-in GitHub Actions artifact when available; the initial evidence was produced locally and is not claimed to have been uploaded to Actions. **No new GitHub Actions workflow has been added** and routine PR/push CI must not train this lab automatically.

**Next discriminating experiment:** impose variable graph branching and limited recovery budgets over unfamiliar structures; evaluate an autonomous neural continuation rather than keeping the teacher in control after one fork. Explicitly include class-balanced linear controls, a hard-coded threshold oracle and an always-relocate comparator. Prioritize independent spatial goals and long-horizon survival, rather than high supervised fork agreement alone.
