# P19 — competence-gated novelty versus continuation collapse

**16 November 2026 · techrote/DRIV3R · existing Draft/HOLD PR #1 exclusively.** External synthetic game-policy research only, not CyberSand native gameplay acceptance.

## Experimental question

P18 found that unconstrained novelty improved descriptor distance while producing 73.6% premature STOP. Can we maintain minimum competence on training/selection worlds while preserving useful variation across the six policy niches?

P19 compared **reward PBT, unconstrained novelty PBT, competence-gated novelty PBT, and random-parent breeding**, with matched initial seeds and TRAIN/selection/held-out world identities within each cohort. Selection uses normalized 0.35 reward + 0.65 novelty as in P18. The **selection-only** gate requires premature STOP <= 50% and reaching the mid-episode goal switch in >= 75% of TRAIN-family selection episodes. A candidate passing the gate ranks ahead of a failing candidate; when no candidate qualifies, selection falls back to reward. Thresholds were fixed using earlier P18 *selection-only* quantiles before running P19 holdouts. Selection never reads P19 held-out outcomes.

Three independent seeds **680, 681, 682** trained **432 distinct scratch-initialized 20-feature/24-hidden-unit MLP checkpoint candidates** with **26,069 actual Adam updates**. Each of six separately trained niches (navigator, branch artist, recoverer, texture worm, material-cautious, chaotic brush) was preserved. Every arm uses initial population 2 + 2 offspring across 2 generations, three TRAIN seeds, two selection seeds, three held-out seeds, and 48-decision worlds. The six synthetic geometry families are divided into four TRAIN and two held-out families, with disjoint train/selection/holdout seed streams. Host-supplied menus enforce legality. PBT parent choices cause subsequent training trajectories to diverge.

## Held-out outcomes

| Arm | Reward proxy | Early STOP | Goal switch reached | Across-niche descriptor distance | Within-niche 2-champion distance | Selection-viable champions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reward PBT | 33.1859 | 26.85% | 80.56% | 0.4346 | 0.1559 | 17/18 |
| Ungated novelty | 14.1437 | **72.22%** | 44.44% | **0.5998** | **0.3903** | 6/18 |
| **Gated novelty** | 30.5359 | 36.11% | 76.85% | 0.4397 | 0.2131 | 17/18 |
| Random ancestry | **33.9868** | **25.00%** | **83.33%** | 0.4428 | 0.2137 | 17/18 |

Gating improved mean reward by **16.39** proxy points and reduced early STOP by **36.11 percentage points** relative to ungated novelty, but **underperformed both reward PBT and random ancestry**. Much of the ungated descriptor diversity vanished when continuation was enforced. Within-niche distance exceeded reward PBT modestly (0.2131 vs 0.1559) but is not evidence of artistic personality or meaningful behavioural generalization.

Across cohorts, mean reward (reward / ungated / gated / random):
- **680:** 33.319 / 16.160 / 32.629 / 35.079
- **681:** 32.269 / 10.545 / 30.388 / 33.414
- **682:** 33.971 / 15.726 / 28.590 / 33.467

**Negative result preserved:** gated novelty failed to beat reward PBT on all three cohorts. No superiority or statistical-significance claim is justified.

## Qualification and exact provenance

Eight local P19 guardrail tests passed. The separate checkpoint-deserialization audit verified all **432 original checkpoint hashes**, independently replayed **144 selected checkpoint evaluations / 864 held-out synthetic episodes / 31,672 host-legal actions**, checked **72 original SVG/JSON trajectory gallery pairs**, and verified **579/579 SHA-256 manifest entries** without mismatches. The independent replay shares the synthetic P0 legal-host/world code, so this is not an independent native CyberSand test.

Source SHA-256 `45115165bc7044345d5c5724c6eec552f9c3ac53a69abdfd5d0fbc56cf23b48a`; test SHA-256 `1fd7a81fdb5ffe007494f04d1816ebb2ae1cb63f7edb9c84f667bfd5b0552136`; replay-audit source SHA-256 `c09861d8ae1d8f35990abcf05dfb2176ca64964e49a97fbd09abda6505150a49`.

Local results SHA-256 `8f95b7a605bfe5687a53d5975b6a54eb7abc5b3055112bfc564889cc1efeaf98`; original full lineage SHA-256 `7be903112f3b181ef4f1b7a80916c712902b691d7ac78542aaaee5db85499b79`; manifest SHA-256 `19308974fbae59dc89a25fbbd7a151b8dc59d6e3efea83c75c55be00d43381d5`. See [P19_EVIDENCE.json](P19_EVIDENCE.json) for split seed hashes, model-side source identity, per-arm outcomes, audit counts and limitations.

Reproduce from `experiments/driv3r/training/population/` with Python 3.11+, NumPy 2.3.x and CPU-only BLAS:

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest test_population_p19_viable_novelty -v
OPENBLAS_NUM_THREADS=1 python population_p19_viable_novelty.py --output /tmp/p19-repro --cohorts 680 681 682 --initial 2 --generations 2 --offspring 2 --training-worlds 3 --selection-worlds 2 --heldout-worlds 3 --horizon 48
OPENBLAS_NUM_THREADS=1 python p19_replay_audit.py /tmp/p19-repro --output /tmp/p19-replay.json
```

**Artifact caveat:** Original full checkpoint weights, per-trajectory SVG/JSON galleries, complete candidate lineage and SHA256SUMS are preserved in a downloadable **local** archive, **not** an immutable GitHub Actions artifact. The committed small ledger is evidence metadata and its original SHA-256 hashes, not a copy of the weights. No single-use workflow, automatic training or paid compute was added. No CyberSand game code, native ABI, Workbench, maps, physics or content was modified.

## Next discriminator

Test competence-constrained **within-niche MAP-Elites archives** across more seeds and a reward/diversity Pareto frontier, not a scalar winner. On P17's three-goal mission, measure actual branch re-entry, retrace recovery and adaptation to material blockages rather than merely surviving to the mid-horizon switch. Compare synthetic action/observation semantics to read-only current CyberSand authoritative host sources before suggesting integration.

**Custody:** Only `techrote/DRIV3R` PR #1, branch `research/import-consolidated-worm-labs`. CyberSand and Neurounner read-only. Draft/HOLD, unmerged.
