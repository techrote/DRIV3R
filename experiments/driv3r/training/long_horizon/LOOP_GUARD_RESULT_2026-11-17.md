# LOOP-GUARD-001 — post-probe recovery conservation and immediate reversal controls

**17 November 2026 · DRIV3R Draft/HOLD PR #1 · synthetic DEVELOPMENT only · no native game mutation.**

## Finding

The earlier SEQUENTIAL-PROBE-001 report attributed weak 128-decision continuation to repeated corridor re-entry. Exact original trajectory inspection instead reveals **short MOVE/BACKTRACK oscillations and avoidable recovery consumption after the first successful escape**. Once cells are painted, returning along the same narrow tunnel is generally not legal; the repeated moves are in *new side routes* and sometimes back to the root, not re-entering the painted tunnel itself.

We restored its **two previously trained 1,033-parameter sequential7 checkpoints (811, 814)**. **No optimizer ran and no new neural checkpoint was trained in this experiment.** We compared the unchanged checkpoint against three deterministic, host-legal inference-side filters operating on its unchanged action probability distribution: `anti_reverse` suppresses an immediate reversal of a MOVE only when another MOVE exists; `conserve` also suppresses BACKTRACK, RELOCATE_START and STOP when there is a free MOVE after a detected trap; `sham_conserve` applies that same rule after the seventh sensor read regardless of whether a trap was detected. All action candidates remain in the original `World.menu(12)`; no actions, observation ABI or World rules are added.

**Result: a model-external recovery-conservation rule markedly improves these synthetic trajectories; a sensor-trigger-specific explanation is falsified by the sham control.** Do **not** claim recovered model competence or a newly trained navigation skill.

## Exact design and split control

- Source: 11 original dependent Python Git blobs independently compared byte-for-byte against the remote DRIV3R PR #1 source tree at inspected head `2c4b3ad020f00376117b01d9bf03d40fbb53d51c`. The entire new source SHA-256 ledger and both original checkpoint SHAs are in the JSON evidence file.
- DEV pilot: seeds 3000/3100/3200/3300, rotations 0 and 1. DEV confirmation: *different* seeds 3001/3101/3201/3301, rotations 2 and 3. Both partitions use the same four **known** development families (`pickets`, `ridges`, `islands`, `zigzag`) and the same two earlier model-training seeds. These partitions are **not** the sealed structural HELDOUT.
- Each partition: 128 matched cap-128 episodes, 16 per kind/arm, two restored checkpoints. All four arms have the identical initial host state and original legal menu, and the same bounded seven-cell sequential probe. The guard is inactive until the stipulated sensor trigger, has no access to oracle kind/viability, and never suppresses the full legal menu.
- The sham is *not an observation budget ablation*: it reads the same seven cells but ignores the obstacle outcome when deciding when to activate the rule. Its equality with `conserve` on traps indicates that the local gain does not require the obstacle value.

## DEV pilot (16 trap episodes / arm)

| Control | Mean new cells in traps | Early STOP | Mean recoveries | Mean new cells in open |
|---|---:|---:|---:|---:|
| Raw sequential7 | 76.625 | 6/16 | 9.000 | 126.375 |
| Anti-reversal | 98.812 | 3/16 | 7.875 | 126.375 |
| **Recovery conservation** | **123.750** | **0/16** | **4.250** | 126.375 |
| Sham recovery conservation | 123.750 | 0/16 | 4.250 | **126.750** |

## Different DEV seeds and rotations (16 trap episodes / arm)

| Control | Mean new cells in traps | Early STOP | Mean recoveries | Mean new cells in open |
|---|---:|---:|---:|---:|
| Raw sequential7 | 86.312 | 8/16 | 9.688 | 122.125 |
| Anti-reversal | 75.125 | 7/16 | 7.750 | 122.125 |
| **Recovery conservation** | **122.875** | **0/16** | **5.125** | 122.125 |
| Sham recovery conservation | 122.875 | 0/16 | 5.125 | **125.750** |

Across both partitions, `conserve` gained **41.844** painted cells per matched trapped episode compared with raw; the paired bootstrap 95% percentile interval was **[25.998, 58.845]**, over 32 paired cases. Early STOP improved from **14/32 to 0/32**. The `anti_reverse` arm did not robustly generalize across the two partitions, whereas conservation did. Sham and true trap-triggered conservation produced identical trapped results, even though they have different activation predicates. This rules out claiming the control needs sensed trap occupancy for these particular matched cases.

## Qualification and provenance

- Four focused tests passed, including same-host-initial-input checks, bounded sensing, legal menu invariants, actual replay of every action/probability/event, deliberate action-distribution tamper rejection and sealed HELDOUT refusal.
- Both runs were replayed end-to-end from original checkpoints: **256/256 episodes**, **30,398** host-legal decisions (15,374 pilot + 15,024 confirmation), and original state checks, neural action probabilities, sensor reports, chosen actions and post-state hashes independently validated. Manifest verification covered **131 files per partition** (excluding the manifest itself).
- Wall time: pilot 17.196s; confirmation 16.432s. Peak RSS approximately 141.0 and 140.4 MiB. Action-selection latency for each arm is in the machine-readable report.
- Original .npz checkpoints and all trajectories retained in **local immutable-path ZIP evidence**, not in Git history. GitHub Actions immutable artifact publication remains outstanding; no new Actions workflow or automatically triggered paid training was introduced.

## Scientific limitations and next discriminating test

The result tests an **external deterministic action filter**, not trained memory, model reasoning or native CyberSand gameplay. The `conserve` filter chooses a fresh MOVE whenever one is available after probing, substantially suppressing optional STOP and recovery. It may fail in worlds where a deliberate backtrack is necessary even though some lateral move remains legal. It can also distort planned motifs and retracing, so it cannot be recommended as a production policy from these fixtures. The controls use the same known development world families and are not structural generalization. `sham_conserve` even improves some open-route cases, making sensor-specific attribution weaker.

Next pre-register a TRAIN-only learned action-ranking preference with matched, deliberately backtrack-required situations and new DEVELOPMENT evaluations that include both *safe* reversals and *necessary* reversals. Retain a legal-action-only guard as a diagnostic upper-bound baseline, preserve motif continuation and recovery efficiency, and open sealed structural HELDOUT only for a precommitted candidate. Native observation ABI changes, if required, remain an interface/handoff proposal only.

**Owner boundary:** Only `experiments/driv3r/training/long_horizon/**` in DRIV3R PR #1. No CyberSand/Neurounner, DRIV3R main, extra PR/branch/workflow or worker-owned file mutation; Draft/HOLD, unmerged.
