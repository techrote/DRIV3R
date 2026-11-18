# DRIV3R Corpus Factory v55 — paired streaming memory contrast

**Date:** 2026-11-18  
**Authority:** `techrote/DRIV3R`, existing `research/import-consolidated-worm-labs` Draft/HOLD PR #1, owned `experiments/driv3r/corpus_factory/**` only.  
**Scientific scope:** synthetic host trajectories for game-policy research; **NOT native CyberSand**, NOT a policy trainer, NOT evidence of an LFM/Sprocket capability improvement.

## Reason for the increment

V54 replaced perpetual historical-ledger replay with event-once streams, but all-decision HELDOUT majority-label accuracy for a memoryless instantaneous-input lookup was still **98.48%**. A system could therefore look successful under aggregate accuracy while failing the events that require memory. V55 exposes a **balanced, independently audit-able contrastive challenge**, without claiming to have authored new host transitions.

## Reproduction / inputs

Use the complete pinned v54 archive, extract its `corpus/` to a directory, then run:

```sh
python3 focus_v55.py --source /path/to/v54/corpus --output /tmp/v55-corpus
python3 test_focus_v55.py --source /path/to/v54/corpus --export /tmp/v55-corpus
```

`v54/corpus/manifest.json` SHA-256: `1ad1c11e8c3d24fd13d0a8aaab8743f8af3a2c1059aa871b6c775b1357fbe107`. V55 manifest SHA-256: `dcd6c65de9732756d43e6538a92837385567f00d577d7cf8235088715427c920`. Generator SHA-256: `d3c9b42a816752e397bc61504579b2f4bcfdc34882c2e3547bd0cdf158dc63d2`. Independent audit SHA-256: `a76ae9551cff6426ea61b7adb182507ada38da0a62c223219961bc54a7b2fe95`.

## Selection and leakage safeguards

1. Read only hash-verified v54 event-stream episode data and evaluation-only provenance. Keep every selected episode **complete and ordered**, including all observation/action history before and after the decisive decision.
2. Hash every instantaneous observation (which carries only the bounded recent eight actions/positions plus a current event delta). Identify all signatures assigned **two different legal actions** under different earlier histories.
3. For every such signature, select exactly one pair of source decisions: same world, different episode, two different labels, TRAIN source loss weights nonzero, maximizing the minimum number of within-episode decisions since the latest new streamed event. Deterministic SHA-order/tie breaking.
4. Keep pair keys, references, labels, world IDs and difficulty tags in `*_EVALUATOR_ONLY` sidecars, not policy inputs. No world identifiers are exposed as features. Require TRAIN/DEV/HELDOUT to retain their original topology grouping. Source world identities must also stay grouped **across** versions v43–v55.
5. TRAIN has `focus_loss_weight` on step records outside `input`: 4 for selected paired decisions and 1 for other originally positive records; original loss-masked repeats stay 0. This is a supplemental curriculum subset and MUST NOT silently replace the broader TRAIN distribution. DEV/HELDOUT have no upweights.
6. Evaluator measures both decisions per pair, the fraction where **both** are correct, 8+/16+ event-age subsets, and material/navigation/mixed classes. A stateless deterministic action function of the identical current observation has a strict 50% upper bound on balanced per-decision accuracy and 0% on both-correct pairs. This is a **structural baseline**, not a learned-model result.

## Qualified outputs

| Split | Selected complete episodes | Stream decisions | Matched contrast pairs | Material | Navigation | Mixed | Age 8+ | Age 16+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 243 | 8,608 | 168 | 64 | 95 | 9 | 28 | 1 |
| DEV | 86 | 3,253 | 59 | 24 | 27 | 8 | 15 | 2 |
| HELDOUT | 80 | 4,192 | 55 | 24 | 25 | 6 | 18 | 4 |
| **Total** | **409** | **16,053** | **282** | **112** | **147** | **23** | **61** | **7** |

All 282 contrasts are same-world, inter-episode pairs of equal instantaneous inputs with opposite labels. The 4,192 HELDOUT decisions contain **110 selected decisive decisions**. A deterministic, stateless signature lookup that predicts one fixed action per signature scores exactly **55/110 = 50%** on the paired challenge, versus the misleading 98.48% ceiling over the broad v54 HELDOUT set.

TRAIN has **2,756** original exact-repeat rows retained for sequential memory propagation but with zero supervised loss. DEV and HELDOUT retain no zero-weight duplicates.

**Determinism and independent audit:** two complete v55 generations were byte-identical across 10 output files. Independent audit compared every selected step to the original source action, observation and original weight; verified legality, full episode boundaries, source identity, split-isolated topology hashes, exact paired observation equality, two opposite legal labels, event ages, absence of labels/metadata in model inputs, loss mask, sidecar hashes and pair counts. All **16,053** selected policy decisions passed. The prediction scorer passed with 110/110 heldout labels under a perfect *teacher-forced replay* input; this is an auditor control, not autonomous competence.

## Interpreting the result carefully

- This work improves **measurement sensitivity** by balancing contradictory observations, not the neural model or the synthetic host. All selected episodes already existed in v54/v53; no new game transitions were created.
- Identical instant observations require some earlier information to predict **the dataset's labels**, but may not prove every navigation action is uniquely optimal (ties or alternate valid routes can exist). Score material pairs separately, and confirm action regret with host dynamics rather than assuming all contrasting labels have an expensive failure consequence.
- Event age is the number of stream decisions since the latest *delivered* event; it does not represent elapsed native-game time. Only 4 HELDOUT pairs have minimum event age of 16+ decisions, so robust long-memory claims are **not** yet justified.
- The scripted prelude is still presented as a bootstrap historical bundle at the first policy step. A symbolic memory controller remains an allowed reference, not the production target. No named SLM trained here, no neural CPU p50/p95 latency/RSS, no native CyberSand evaluation.
- The derived corpus is published as source + exact hash manifest in Git and a reproducible release archive; immutable GitHub Actions artifacts remain a separate publication gate. CI should remain lightweight and opt-in for full corpora.

## Consumer experiment

Use pinned **LFM350/Sprocket target weights** only when legitimately available. Train/fit adapters on full ordered v54 TRAIN episodes, with v55 TRAIN as a supplementary critical-step curriculum. Compare a recurrent/state-carrying model to the **same** model reset every decision. On DEV, then once on sealed HELDOUT, report both-correct pair rate, balanced critical accuracy, 8+/16+ event-age results, material-calibration success and whole-episode *synthetic-host* terminal success (if actually executed). Report CPU latency (p50/p95) and RSS plus exact checkpoint and tokenizer hashes. Do not conflate teacher-forced scoring with closed-loop autonomy. Native CyberSand is a distinct subsequent validation step requiring ABI verification and no CyberSand repo writes.

## Ownership

No paid API/GPU, no new branch/PR/issue, no edits to CyberSand, Neurounner, other DRIV3R labs, workflows or main. DRIV3R PR #1 must remain Draft/HOLD. Large generated shards are **not committed** to Git.
