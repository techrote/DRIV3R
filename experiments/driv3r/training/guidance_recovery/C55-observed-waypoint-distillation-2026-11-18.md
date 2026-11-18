# DRIV3R C55 — Host-Observed Waypoint Distillation (18 November 2026)

**Scientific decision: RESEARCH HOLD; do not promote.** This bounded CPU experiment establishes an improved *narrow, constructed, already-observed reachable-waypoint* task, not general recovery from novel remote goals or native CyberSand gameplay. C54's independently generated far-goal Wilson/Kruskal challenge yielded zero goals for all controllers; C55 deliberately tests an easier, independently specified task to diagnose where missing information matters. Do not combine success rates across these task distributions.

## Provenance and immutable boundary

- Repository/branch: `techrote/DRIV3R`, existing Draft/HOLD PR #1, `research/import-consolidated-worm-labs`. All research publication restricted to `experiments/driv3r/training/guidance_recovery/**`.
- Frozen scientific dependencies: C54/C53/C52 and the older local synthetic host, exactly as retained in the attached original C54 archive. Authoritative CyberSand native interface remains independent and unchanged; old `d_common` is a fixture only. No modifications to CyberSand or Neurounner.
- The host produces legal action menus, executes every selected action, retains the state/RNG clock across a real synthetic goal change, and revalidates all actions with `World.apply()`. A separate pre-existing 24-step graph construction prefix was host-executed in the upstream generator, but C55 does not reconstruct its per-step action menus. From that pre-existing fixture, C55 records new 9/19-tick execution prefixes and locally observed neighbours.
- C55 goal constructor selects a **not-yet-visited target already offered as a legal frontier neighbour of some actually observed node**, with an observed traversable route of 2–14 steps. This target is handed to every policy as the same synthetic host-issued new goal. No hidden map is read by the deployed policy or by its planner. This is not a general randomly placed new goal.
- Guide arms: no advice, signed weak backtrack (0.35), signed strong backtrack (1.0); guidance expires after 9 turns and does not authorize any forbidden action. Agent policy remains stochastic. Every episode includes matched RESET, ROUTE, ALIGNED, graph-shuffled-label and random-preference controls.

## Actual optimizer and source-matched evidence

The learned policy distills **an observed graph's next legal waypoint** into a 13-feature action scorer, rather than fitting sparse full-goal winner labels. A small `13 → 20 → 1` tanh head was trained for 14 CPU AdamW epochs, **630 optimizer updates per head**, with graph-level outcome-label shuffling for the matched negative-control model. There were **2,862 legal-candidate examples**, **1,298 positive examples**, **358 distinct supervised state hashes** from **30 contributing training graphs** (48 generated graph identities). A rerun independently reproduced byte-identical safetensors for both heads. This is dense **route imitation**, not reinforcement learning or optimization of whole-goal return. The network sees both the observed next waypoint and the candidate action; learning this conditional ranking is a deliberately narrow capability.

A separate held-out **8-graph unseen-family blind action-ranking** test yielded top-1 route-following precision of **96/100** for ALIGNED and **66/100** for SHUFFLED. Those are state-level results with correlated states per graph, not 100 independent graphs, and do not by themselves qualify gameplay.

A 32-graph online preview was inspected before the larger replication. The preview yielded 36 goals for ALIGNED, 30 for RESET, 31 for SHUFFLED and 44 for the non-neural ROUTE. The ensuing frozen-checkpoint validation used fresh seeds with **128 different graph identities** (64 Wilson/Kruskal and 64 fresh familiar-family), excluded **160 graph identities from prior C54/C55 source cohorts**, and made **no training or threshold change**. This larger evaluation is independent at graph level, but the research sequence as a whole is exploratory, not preregistered confirmatory.

## Frozen-checkpoint external validation

**264 matched guided synthetic episodes per controller; 132 no-guidance episodes per controller; 396 matched goal-change/arm records; 128 generated unseen graph identities, of which only the eligible states contribute outcomes.**

| Controller | Completed goals / 264 | Condition-level regressions vs RESET | Wilson | Kruskal | Familiar families |
|---|---:|---:|---:|---:|---:|
| C18-A0 | 160 | 26 | 48 | 34 | 78 |
| C34-RESET | 199 | 0 | 50 | 47 | 102 |
| C55-ROUTE, explicit memory heuristic | **226** | 5 | **53** | **62** | **111** |
| C55-ALIGNED, trained route imitation | **220** | **4** | 50 | 59 | 111 |
| C55-SHUFFLED, trained wrong labels | 193 | 7 | 46 | 46 | 101 |
| C55-RANDOM, matched bounded bias | 191 | 14 | 46 | 45 | 100 |

ALIGNED minus RESET: **+21 completed goals**, from 20 favorable and 5 unfavorable graph clusters, exploratory graph-bootstrap 95% range about **+8 to +35** goals. ALIGNED minus SHUFFLED: **+27** goals, exploratory 95% range **+12 to +42**. ALIGNED minus explicit ROUTE: **−6**, exploratory 95% range **−18 to +5**. These intervals do not adjust for the iterative research history. Improvements on independent unseen-family **Wilson** were exactly zero versus RESET (50 vs 50); **Kruskal** gained 12 (59 vs 47), demonstrating heterogeneous transfer rather than universal generalization. The strong handcrafted ROUTE controller still outperformed the neural head overall.

ALIGNED retains **four condition-level regressions** against RESET, and ROUTE retains five. These fail the hard no-regression gate; no promotion is warranted. Every policy separately executed the no-guidance arm (no cloning or copying of baseline outcomes), maintaining **132/132 matched no-guidance action+painting hashes** per policy, zero executed synthetic-host illegal rejections, and identical protected-cell hashes. The first, faster evaluation preview was replaced by this fully executed parity check before final custody. These constraints do **not** establish whole-trajectory personality preservation under guidance: preferred/opposed ink-turn counts shift and require independent blinded personality qualification. No native CyberSand actor, Workbench or authoritative ABI was executed.

## Reproduction and custody

- `c55_observed_waypoint_distillation.py`: frozen source and CPU trainer/evaluator. Inputs require the exact old C54 archive prerequisites and independent Wilson/Kruskal world factories; no game changes.
- `c55_external_validation.py`: frozen-model fresh graph validation; source-level cohort exclusion; no retraining.
- `c55_curated_gallery.py` and `c55_gallery.py`: recreate 27 actual action-hash-matched traces in nine matched graph triplets (successes and regressions). Gallery panels distinguish the C55 new prefix from the older source-generated fixture path and show individual action-hash prefixes.
- `test_c55_observed_waypoint.py`: six synthetic-host and safety regression tests.
- `C55-evidence-ledger.zlib.b64` and `C55-extract-ledger.py`: full training, blind, online preview, external validation, paired outcomes and curated gallery trace records; uncompress with `python C55-extract-ledger.py C55-evidence-ledger.zlib.b64 OUTPUT_DIR`. Each embedded JSON document is SHA-256 verified by its manifest. The JSON evidence includes all graph identities and paired per-policy action/painting hashes.
- `C55-compact-summary.json`: source/checkpoint hashes, full qualification metrics and self-describing trial limitations.

**Checkpoint custody:** CPU-trained model binaries are deliberately **not committed to Git**. Their exact SHA-256 digests appear in the compact summary and are in the attached local owner-evaluation archive; immutable GitHub Actions artifact publication remains **OUTSTANDING**, not silently asserted. The local archive also includes source, full ledger, both safetensors, frozen C54 prerequisite archive, test logs and manifest. Python tests and optimizer refit passed. GitHub exact-head CI, if available, is a source-smoke test only, not proof of physical or statistical reproducibility.

**Next bounded experiment:** test a far-goal scenario with explicitly limited remembered graph routes and a higher observation budget, separating success-floor effects from route-memory advantage; independently measure guidance's effect on personality on matched trajectory families. Do not treat improved classification alone as a solution to previously unsolved remote-goal recovery.
