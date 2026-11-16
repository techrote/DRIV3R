# W25 — independent exploratory material/frontier retention (16 November 2026)

**Status: local CPU research completed, NOT source-qualified hosted training; Draft/HOLD; no promotion.**
**Scope:** solely SPROCKET auxiliary neural fusion policy; rooted in `kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668` lineage, but **zero 500M root forwards, root optimizer steps or LoRA updates**.

## Admitted source and experimental contract

- Input: original W24 GitHub Actions artifact preserved in the research handoff (archive SHA-256 `c1dde980b6eb2b9e2979a7f84168ff7c0e7a3a04529a94538e4c9b37375dd0b6`). No artifact upload/migration to DRIV3R Actions is implied.
- Replayed original external W0 source SHA-256 `81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e`, with reconstructed W18 material/protection mask and W22 seeded C-pockets / baffle fixture. Not CyberSand native physics or its authoritative action ABI.
- Frozen W24 wave-1 parent SHA-256 `2c9f46abc8e115370f0f6307b252a5c43a3dab258810df297c6b89ab652ccca2`; frozen wave-3 parent SHA-256 `3cee67731f724f840af7d009c47772e523e2d599d88c8f4824af25b8c543c86a`.
- Train seeds: 11003/11011/11027/11047; validation: 12007/12011/12037; untouched test: 13001/13009/13031. Families: greebles, pipework, architecture.
- 6,126 observed training states; 4,603 validation states; 4,597 independent test states. 1,900 local TRAIN material counterfactual pairs; 1,500 validation and 1,500 test pairs. Same-observation patched material pairs do **not** establish world-persistent physical causality.
- Three actual 120-step AdamW waves (360 total), training only eight fusion head tensors; frontier balance, paired-material counterfactual rescue, train-seed on-policy retention. Original frozen W24 experts preserved. No optimizer restart between waves. Local torch CPU, one thread; 32-second first execution.
- Two independently launched reproductions returned identical canonical JSON for complete loss histories, held-out trajectories and summary metrics (excluding elapsed runtime); note this is local software replay, **not** Actions artifact custody or independently reproduced native behaviour.

## Frozen held-out 128-decision trajectories

Nine world/family combinations, 128 applied decisions each for learned policies:

| Arm | Mean distinct cells | Branches | Retraces | Both targets right / 1,500 material pairs |
|---|---:|---:|---:|---:|
| W24 wave 1 (frozen) | 105.44 | 69 | 82 | 249 |
| W24 wave 3 (frozen) | 104.78 | 62 | 71 | 239 |
| W25 wave 1 | **112.44** | 59 | 58 | **254** |
| W25 wave 2 | 113.44 | 55 | **30** | 252 |
| W25 wave 3 | 105.44 | 50 | 63 | 251 |
| Random legal | 58.33 | 130 | 258 | not measured |
| Material teacher | 49.33 | 15 | 344 | not measured |

Wave 1 was selected on **validation data only**, not test data. The test 128-step threshold of material >=95% of W24 wave1 and spatial >=103% of W24 wave3 was met, but the validation combined retention score was only 0.911; this is not a robust prospective success. The 128-cell average gain vs W24 wave3 was +7.67 cells, **95% paired bootstrap interval -1.89 to +17.89** across only nine world/family clusters. Five improved and four regressed; no statistically established general improvement.

## Independent 256-decision horizon (materially negative)

| Arm | Mean distinct cells | Retraces |
|---|---:|---:|
| W24 wave 1 | 163.89 | 382 |
| W24 wave 3 | **182.11** | 288 |
| W25 selected wave 1 | 171.89 | 332 |
| W25 selected sampled T=0.7 | 189.89 | 225 |
| Material teacher | 64.33 | 855 |

The selected greedy checkpoint **regresses by 10.22 distinct cells** versus W24 wave 3 at the longer horizon. Paired 95% bootstrap interval: -43.67 to +19.78; the sampled arm is exploratory and cannot rescue the unproven greedy policy. All 128/256 closed-loop actions selected by these policies were host-legal with no synthetic protected-cell violations, but this says nothing about native CyberSand correctness.

## Checkpoint custody and qualification gap

Three locally saved model+AdamW+RNG checkpoints exist at 120/240/360 steps with SHA-256:
- wave 1: `46c5c915bec4155ca73928aa62e33371506e40d6368c153dfb9b3b7e6986f6e5`
- wave 2: `7c23047319fefd94494f5ede49bb45589fcf5ac328ab5c99c2e2c869e9b8781b`
- wave 3: `5b1a57d8118dcfdbc1f592e7b6e8910a9e7edb6074810927593f5c80d6f337fa`

Local standalone trainer source SHA-256 `4d431e543457d9edc6492b48d1c5e56e524d4736c27fc8107f61094579727b85`. Local evidence ZIP SHA-256 `03773f976fe6720b115b1fca04aa65d0601545242c6ac6898247e9c942b03c93` contains source, checkpoints, complete trajectory JSON, gallery, loss and bootstrap receipts. **Exact W25 Python source is now committed in this owned lane as a reversible, SHA-256-checked gzip/base64 transport wrapper**: [`worm_train_v25.py`](worm_train_v25.py). Run `python worm_train_v25.py --print-source > /tmp/worm_train_v25_readable.py` to inspect the uncompressed original. Independent base64 Adler-32 verification against the source emitter passed (`9904f6bc`), and the patched exact source was rerun locally with unchanged loss, trajectory and metrics JSON. This archival wrapper is deliberately temporary; replace it with the decoded readable Python source in the same file after independent inspection. **The checkpoint ZIP has not been published as a DRIV3R Actions artifact**, so no hosted exact-source reproducibility claim is made. Publish bounded opt-in Actions checkpoints, without adding automatic training, before advancing beyond an exploratory result. Do not commit checkpoint binaries to Git. No paid external compute was used.

## Decision

**RETAIN as negative/diagnostic local research, NOT promote.** The small 128-step improvement disappears at 256 steps and paired uncertainty is wide. Favor longer-horizon, seed-clustered gates and predeclared material-retention thresholds. Root Sprocket, d1 and tiny model comparisons remain unperformed. No writes to CyberSand, Neurounner, other DRIV3R laboratories, DRIV3R main or production provider settings.
