# SENSE-RANGE-001 — bounded occupancy sensing versus strict host-menu aliases

**Date:** 17 November 2026. **Repository:** `techrote/DRIV3R`, **only** research PR #1 / `research/import-consolidated-worm-labs`. **Status:** DEVELOPMENT-only positive local sensor result; no native or sealed-HELDOUT qualification; Draft/HOLD.

## Question and source provenance

Earlier MENU-ALIAS-001 established a strict alias: `far_trap` and `open` expose an identical synthetic host-legal action menu and identical existing ranker features, although the correct initial decisions are opposite. Could a bounded, read-only occupancy observation remove this alias without changing movement/paint permissions, reward, model capacity or optimizer budget?

All reused dependencies match Git blob identities from the remote long-horizon laboratory at inspected branch HEAD `4c8448d045e61e5888844b04a0b298dcfe5879c9` before this work. `sensing_probe_lab.py` and `test_sensing_probe_lab.py` are new; no other existing source changed. Source identity and complete checkpoint SHA-256 ledgers are in `SENSING_RANGE_EVIDENCE_2026-11-17.json` and the two immutable local campaign manifests.

## Controlled design

- All four policies received identical synthetic legal menus, world geometry, explicit goals and maximum **128** autonomous decisions per episode. `frozen` is an unadapted 961-parameter goal+recovery ranker. Three matched **1,009-parameter** rankers were initialized from its parameters and trained for equal optimizer budgets, but received different observation radii: `blind` (0), `range3` (3 cells), `range7` (7 cells).
- The two extra inputs per **MOVE** action are a bounded cardinal ray occupancy flag and the normalized contiguous free-cell count. The sensor reads only cells at distances 1..radius from the current cursor. No hidden target/viability label, no oracle path, no fabricated action, and no mutation of the host's legal menu is permitted. `BACKTRACK`, synthetic `RELOCATE_START` and STOP receive zero additional ray features.
- **This sensor is a research-only proposed observation, NOT part of CyberSand's verified native contract.** Synthetic terrain is visible to the external research harness; granting it to the trained model is the experimental intervention, and says nothing about what native agents currently observe.
- Each seed used a genuine 36 → 72 → 128 curriculum, 18 pretrained optimizer updates per stage, 48 subsequent Adam updates per stage for every adapted arm, 64 matched natural samples per stage and 48 specially constructed training states per stage (144 per run over three stages). Pre-existing natural-feature samples lack ray data and are explicitly **zero-padded** for all arms; this is a potential domain-shift limitation. All optimization data came from TRAIN and no DEVELOPMENT labels or outcomes were used in optimization.
- Independent training seeds **711 and 714** (712 and 713 were rejected before optimization because the geometry viability certificate could not be constructed on all TRAIN orientations; they are not counted as completed seeds). DEVELOPMENT uses four known non-heldout geometry families, two orientations, and seed 791: eight matched environments per situation per seed; 96 episodes per seed and 192 total. This is an interpolation test within previously studied family templates, not blind structural transfer.

## Development results — 16 episodes per arm and situation

| Controller | Correct initial choice | Early STOP | Mean new painted cells |
|---|---:|---:|---:|
| ****Far hidden trap**** | | | |
| frozen | 0/16 | 16/16 | 5.00 |
| blind | 16/16 | 7/16 | 100.94 |
| range3 | 15/16 | 12/16 | 51.19 |
| range7 | 16/16 | 4/16 | 109.06 |
| ****Open route**** | | | |
| frozen | 16/16 | 8/16 | 99.75 |
| blind | 0/16 | 7/16 | 98.81 |
| range3 | 1/16 | 12/16 | 52.00 |
| range7 | 16/16 | 8/16 | 100.31 |
| ****Near visible trap**** | | | |
| frozen | 0/16 | 16/16 | 3.00 |
| blind | 16/16 | 8/16 | 95.88 |
| range3 | 16/16 | 12/16 | 53.94 |
| range7 | 16/16 | 4/16 | 109.06 |

**Decisive finding:** on the matched far-trap and open-route pairs, the `range7` policy chose the correct first action in **32/32** cases. The blind adapted policy always chose retreat (**16/16 far traps; 0/16 open**); `range3` did nearly the same (15/16 traps; 1/16 open). All sensors retained identical host-legal action menus. Range 3 cannot distinguish far-trap from open at the fork, whereas range 7 changes only the bounded local occupancy observation and distinguishes the two in every certified pair.

Across the same far/open families, the `range7` model painted mean **109.06** cells per far-trap episode and **100.31** per open episode, versus blind's **100.94** / **98.81**, respectively. This does **not** prove general long-horizon planning: these matched synthetic forks are intentionally simple, and the learned ranker can exploit the added range feature.

### Sensor-removal intervention

The trained `range7` weights were evaluated again at the original forks with both additional observation inputs suppressed, **without retraining**. Seed 711 then selected correctly in 4/8 far and 4/8 open cases, versus 8/8 each with range 7. Seed 714 remained correct in 8/8 far cases but fell to 0/8 open, versus 8/8 each with range 7. This establishes reliance on the supplementary feature at least in the open/trap discrimination; behavior is seed dependent under feature ablation. The exact per-action probabilities and identities are retained in each campaign's `mask_ablation.json`.

## Qualification and evidence custody

- **Focused tests: 4/4 passed.** Includes exact menu equality, observation-alias proof, TRAIN-only row replay and tamper rejection, architecture/restore check, short real optimizer training and full quick evidence replay.
- **Both complete campaigns reverified successfully** in a separate process, reconstructing the original checkpoint weights and all neural action probabilities, applying all 192 original host-legal synthetic episodes (14,177 decisions), replaying all 288 TRAIN-case provenance rows, and verifying **128 SHA-256-manifested files per seed**. A memoized factory was used only to avoid rebuilding identical source states during this independent replay; each episode received a deep copy of the source state.
- Checkpoints: 8 real `.npz` files (two seeds × four policies). Exact SHA-256, parameter counts and update counts are recorded in the machine-readable ledger. Campaign CPU wall times were **25.385 s** and **27.415 s**, with reported peak process RSS **124,412 and 124,548 KiB**. Mean per-decision neural latency was approximately **0.098–0.123 ms**, varying by model and episode type. All measurements are specific to this local CPU environment.
- The 192 episode summaries include first-action distributions, initial/final state hashes, full trace file SHA-256 values and outcomes. Original full source-replayed JSON traces and checkpoints are retained in local ZIP evidence, **not** placed in Git history. GitHub Actions immutable-artifact publication is **outstanding**; no new workflow or paid compute was initiated.
- This source import and synthetic evidence do **not** certify a native CyberSand-compatible observation API, real game actor transfer or structural heldout generalization. No native acceptance evidence exists, and the sealed structural HELDOUT remains unopened.

## Interpretation / decision

This experiment isolates a real information bottleneck: the existing memoryless feature vector provably aliases a recoverable hidden dead end and an open route. A bounded seven-cell occupancy observation breaks that alias and allows 1,009-parameter policies to learn the correct initial action on every matched DEVELOPMENT pair here; a three-cell ablation cannot reliably do so. It is **an observation-sufficiency result in the synthetic world**, not an approved native sensor or a claim of general recovery planning.

**Next permitted research:** test alternative observation forms with the same information budget (time-sequenced legal probes versus static range, persistent state under equal menus), evaluate on new TRAIN/DEVELOPMENT fixture types and unmodified ordinary autonomous episodes, then separately propose any game-side sensor contract without implementing CyberSand changes. Do not tune repeatedly on the sealed heldout. Keep PR #1 Draft/HOLD and do not merge.
