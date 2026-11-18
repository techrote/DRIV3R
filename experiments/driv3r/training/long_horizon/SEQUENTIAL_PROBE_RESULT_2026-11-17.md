# SEQUENTIAL-PROBE-001 — seven-cell static occupancy versus movement-based sequential probing

**17 November 2026 · DRIV3R research PR #1 · Draft/HOLD · synthetic DEVELOPMENT only.**

## Question and result

The earlier SENSE-RANGE-001 result demonstrated that a non-native seven-cell occupancy ray separated an otherwise identical hidden dead end and open route at the first decision. Can an agent with the **same seven occupancy-cell inspections**, distributed across three physical decisions, recover comparable performance without a long-range initial view?

**Outcome: locally successful conditional escape, worse overall long-horizon robustness.** A trained sequential policy moved three real cells into the corridor, recognized the obstacle, then used host-legal BACKTRACK to recover. Across two seeds it reversed at the third probe in **16/16 traps**, and proceeded forward in **16/16 open routes**. Removing the third probe's report and its associated bounded memory reversed the trapped policy's immediate decision in **16/16 cases**, but did not change the open decision (0/16). However, the same sequential policy prematurely STOPped in **6/16** trap episodes versus **2/16** for the static seven-cell controller, with mean post-fork painting **76.625 versus 113.312** cells. The failure is highly seed-dependent: six early STOPs came from seed 811; seed 814 had none for sequential sensing.

**Interpretation:** seven distributed reads can suffice for *local recognition and reversal when there is remaining recovery budget*, but this experiment does **not** establish that sequential probing matches the static sensor for 128-decision gameplay. The local positive observation result coexists with negative long-horizon performance.

## Controlled setup

- **Exact source baseline inspected before edits:** remote branch `research/import-consolidated-worm-labs` at `96ab90145b85b031a9d156293fe73d3520b632f7`. Local reused dependency Git blob identities matched all ten remote file identities. Source SHA-256 ledger is in `SEQUENTIAL_PROBE_EVIDENCE_2026-11-17.json`.
- Host world: only original synthetic `World.create`, `World.menu(12)`, and `World.apply` actions. A newly authored TRAIN/DEVELOPMENT corridor uses static SOLID/protected material with an open/trap obstacle at six cells ahead, rotated in all four cardinal orientations **before** any actions. The sole initial host move places the cursor one cell inside the corridor. No position, ink, recovery count or legal menu is fabricated during gameplay. The first host menu and original model feature tensor are exactly identical across trap/open pairs. Trapped states have recovery headroom, unlike the earlier exhausted-recovery fork; moving forward and reversing are both physically legal in the synthetic World.
- **Observation intervention is model-only and non-native.** Both adapted 1,033-parameter rankers have identical architecture, feature dimension and matched training/optimizer budgets, but `static7` reads seven occupancy cells along the forward ray at the fork; `sequential7` reads one at the fork, three after the first actual MOVE, and three after the third actual MOVE (1+3+3). These are **seven cell inspections**, not seven unique cell coordinates—overlap is allowed and the probe physically pays for relocation. The blind control uses zero reads but the same-sized ranker. All neural choices are taken from actual host menus. Memory of a discovered obstacle is a bounded, observation-derived research feature, not native agent state.
- Two independent TRAIN seeds **811 and 814** each use a 36→72→128 decision curriculum: 12 baseline Adam updates per stage, 45 adaptation updates per stage per arm, 48 randomly sampled natural TRAIN records and 320 paired synthetic TRAIN action records per stage **per arm**. The original natural records have no extra sensor values, so those inputs are zero-padded (a documented domain-shift risk). Targets, labels and world construction are strictly TRAIN-only. No DEVELOPMENT outcome is used to change weights or labels.
- DEVELOPMENT consists of the four known synthetic geometry families `pickets`, `ridges`, `islands`, `zigzag`, two orientations (0 and 1), two paired corridor types, three policies and two seeds: 96 **128-decision-cap** episodes in total. This is not a structurally new/sealed HELDOUT evaluation. The aligned initial host menus are unchanged across all controllers; the sensor is not an approved CyberSand ABI.

## DEVELOPMENT: two seeds aggregated (16 episodes per situation/controller)

| Condition | Blind (0 reads) | Static 7 (7 reads upfront) | Sequential 7 (1+3+3 reads) |
|---|---:|---:|---:|
| Trapped: initial goal-forward MOVE | 16/16 | **0/16** | 16/16 (probe deliberately enters) |
| Trapped: eventually return to root | 16/16 | 16/16 | 16/16 |
| Trapped: mean decisions to first root return | 11 | **1** | 7 |
| Trapped: premature STOP | **1/16** | 2/16 | **6/16** |
| Trapped: mean new painted cells | **116.875** | 113.312 | **76.625** |
| Open: initial goal-forward MOVE | 16/16 | 16/16 | 16/16 |
| Open: pass beyond former obstacle site | 16/16 | 16/16 | 16/16 |
| Open: premature STOP | 0/16 | 0/16 | 0/16 |
| Open: mean new painted cells | 126.688 | 125.688 | 126.375 |

**Ablation:** On the same third-probe pre-action world state, the trained sequential model switches from `BACKTRACK` to a forward action in **16/16 traps** when the three-cell reading and derived retreat-memory feature are jointly masked. The same intervention changes no first post-probe action in 16 open cases. This proves causal dependency on the *combined* observation/memory channel at that decision, not independent reliance on the raw ray versus memory. Nor does it prove that probe motion is better than unsensed movement plus eventual host-menu obstruction.

**Delayed failure:** In the 811 campaign, the sequential policy consumed recovery budget through repeated later revisits and stopped after only 16 decisions on six of its eight trap episodes. The 814 campaign continued to budget in all eight traps. This large variance prevents a stable generalization claim. In the open conditions, all controllers continued to the 128-decision cap. The blind comparator also eventually backtracked from every trap, so mere eventual root return is *not* a sufficient success metric; timing and painted-cell continuation are reported separately.

## Reproducibility, source identity and custody

- Focused test suite: **4/4 passed** (`test_sequential_probe_lab.py`), covering identical original menus and features, seven-read accounting, recoverability, TRAIN-only record and exact movement/sensor replay, matched parameter counts, checkpoint save/restore, immutable evidence directories, synthetic optimizer execution, and deliberate trace-tamper rejection.
- Both real campaigns: **6 trained `.npz` checkpoints**, each 1,033 parameters, full Adam optimizer stages, checkpoint restored before evaluation; **96 original replayed episodes, 11,356 host-legal decisions, 5,760 TRAIN ledger rows**; state, action probabilities, material edits, sensor counts, and corresponding SHA-256 file manifests verified. The original detailed traces are retained locally as immutable evidence directories and accompanying ZIP; no binary checkpoint is committed to Git history.
- Seed 811: 48 episodes, 5,472 decisions, 2,880 TRAIN ledger records, 21.022s CPU wall, peak RSS 132,428 KiB. Seed 814: 48 episodes, 5,884 decisions, 2,880 TRAIN ledger records, 21.553s wall, peak RSS 131,392 KiB. Neural choice mean latency in aggregate groups was approximately **0.171–0.194 ms**. Environment-specific measurements, not physical native agent timing.
- Both source-matched archives contain a per-file `SHA256SUMS.json` verified by independent exact-episode and TRAIN source reconstruction. Machine-readable checkpoint hashes, all 96 trace SHA-256 values, action distributions, latency and read budgets are committed in `SEQUENTIAL_PROBE_EVIDENCE_2026-11-17.json`. The `SEQUENTIAL_PROBE_TRAJECTORIES_2026-11-17.svg` gallery is drawn from original synthetic legal event traces, not a generated gameplay screenshot.
- **Gaps:** The original checkpoints are retained in local evidence, **not yet immutable GitHub Actions artifacts**. No additional workflow was introduced. The sealed structural HELDOUT was not opened; no native CyberSand observations, actor moves or Workbench acceptance occurred; no paid compute was used.

## Decision and next experiment

**Do not promote** sequential probing above static sensing or the original frozen baselines. The objective is not improving numeric one-step discrimination at the cost of later continuity. Next test should isolate **post-recovery loop avoidance** using an independent 128-decision loss and original natural world controls. A sensor-only versus memory-only ablation and a first-failure control for repeated corridor re-entry would separate the initial detection benefit from the later budget-collapse mechanism, with sealed HELDOUT still reserved for a precommitted qualifying candidate.

This evidence belongs solely to `techrote/DRIV3R` PR #1, Draft/HOLD. CyberSand and Neurounner are read-only; the historical `d_common` fixture is untouched. Do not merge, mark Ready or create experiment-specific PRs/workflows.
