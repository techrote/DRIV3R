# FT-021 — matched training residency and shifted held-out policy probes

**Date:** 16 November 2026. **Disposition:** original local bounded CPU experiment, source/tests committed solely to DRIV3R Draft/HOLD PR #1; **no hosted Actions training qualification or immutable Actions artifact yet**. Do not infer CyberSand-native gameplay acceptance, SLM capability, or historical FT-019 replay.

## Research design and lease custody

- Source ft021_training_residency.py SHA-256: 33fa099598718d54a6e39ace8dd3600361d0bdcbda5df92b4d15667c5b609680; Git blob: e3c7353abcdf01d78c09b0ec7b65c0769e35638b.
- Tests test_ft021_training_residency.py SHA-256: 1b19fef70320641997c1138184793d06f84f753e498e6fad5115fc7ec157f753; Git blob: d48bc76b4a60d9d52dbd67c17229b4de46739488.
- Four distinct original FT-020 Adam-trained 20×24×1 NumPy policy checkpoint files recovered from the original **local** FT-020 ZIP; exact SHA-256 anchored in source PARENTS and independently verified before training. Not an immutable GitHub Actions checkpoint archive.
- Each model receives a distinct synthetic shifted curriculum with deterministic disjoint train/held-out RNG seeds; 192 training one-step menus, 128 held-out menus per niche. Held-out menus use **70 rather than 45 synthetic obstacles** and a 75% rather than 40% randomly assigned goal-switched feature. The exact world/feature generator is in the source.
- Training is an actual optimizer run with **fresh Adam moments initialized from frozen FT-020 weights**. It does **not** resume the original Adam optimizer state. Four niches have independent model input/checkpoint hashes and unique worker leases.
- Compare **four processes with one model each** against **two processes with two models each**. Each residency mode trains the same four source checkpoints with exactly the same deterministic update examples. Model outputs must be byte-identical across modes. Experiments: 150 and 900 updates/model/mode, totaling **8,400 actual Adam updates**, 16 trained outputs (eight unique child checkpoints).
- Explicit invocation only: no push-triggered training, new workflows, paid compute, game/Workbench changes, or writes to CyberSand/Neurounner. Hard ceiling 1,200 updates/model, 120 s/mode, 450,000 KiB sampled RSS/process, 1,000,000 KiB summed sampled peaks. Child failures preserve stderr/stdout and FAILED.json where possible.

## Measured local CPU performance

| Workload | Four processes, 1 model/process | Two processes, 2 models/process | Four-vs-two relative rate | Sampled sum RSS: four / two |
|---|---:|---:|---:|---:|
| 150 updates/model/mode | **1.007592 s** | 1.143663 s | **1.135×** | 392,084 / 196,128 KiB |
| 900 updates/model/mode | 1.313264 s | **1.161988 s** | **0.885×** | 392,044 / 196,048 KiB |

Wall times include child process spawn, Python/NumPy imports, data construction, real optimizer updates, held-out scoring and result writing. The long trial favours dual-resident processes; short trial favours four single-model processes. These opposite signs, with only one trial at each workload and reversed order not fully replicated, **are not statistically robust runner-selection evidence**. RSS values are sums of separately sampled worker peaks, not a concurrent aggregate peak. Four worker processes are launched before joining; simultaneous CPU-tick advancement was not independently established. This is a local Linux container result, not GitHub-hosted CPU performance.

**Exact-output parity:** all four child checkpoint SHA-256s and all four held-out reports match exactly between one-resident and two-resident arrangements at both workloads. Source enforces collision detection and exact checkpoint hashes; 4/4 focused local tests passed, including actual child-process dispatch.

## Shifted held-out learning: before → 150 updates → 900 updates

All figures use 128 independent synthetic menus per policy; action-choice accuracy against the shifted synthetic teacher, **not physical game success**.

| Niche | Top-choice accuracy before | After 150 | After 900 | Cross-entropy before → 900 |
|---|---:|---:|---:|---:|
| Navigator | 85.16% | 92.19% | 89.84% | 0.4861 → 0.3652 |
| Recoverer | 78.13% | 92.97% | 92.19% | 0.8371 → 0.3909 |
| Branch artist | 56.25% | 94.53% | 95.31% | 0.9566 → 0.6679 |
| Material cautious | 84.38% | **77.34% (regressed)** | 84.38% | 0.5800 → 0.4521 |

The material-cautious accuracy regression after 150 updates is retained as a negative result. Cross-entropy improved in all four 900-update tests, but the objective is a constructed one-step synthetic teacher. No long-horizon trajectories, replay legality, painted-cell outcomes, multi-goal completions or native CyberSand host-action ABI validation are established by FT-021. The harness's synthetic RETRACE menu item may not represent a genuine parent/retrace action and **must not be passed off as canonical host-legal gameplay**.

## Source-matched local evidence identities

- 150-updates original report SHA-256: 42f5c95aa67f8179bd225faf4ba708a8485fced168cfccdb498bd22994a02800
- 900-updates original report SHA-256: 2c85071e327b8d25caccfb84782a21e1c0e424541d6e9fba03415e4bc3165ea6
- Locally archived ZIP SHA-256: 719d651596d03f60a54d6a557dbd7111ba7cd6ffca46d328345770c81f30cdfb
- Original checkpoint ZIP remains outside Git, as do all FT-021 child checkpoints. The ZIP is local/session evidence only; a hosted artifact-custody gate is still open.
- No failed process was observed in the two measured trials. Failure handling was unit-tested on invalid dispatch; full timeout/kill and RSS-cap fault injection are not yet independently qualified.

## Follow-on experiment decision

Do **not** promote the long-trial dual-resident configuration as universally faster. Repeat modes in counterbalanced order and on the same bounded GitHub-hosted CPU runner via an *existing* opt-in mechanism, if available, with immutable training artifacts. Then evaluate policy usefulness on legal sequential synthetic/replayed CyberSand-compatible worlds, enforcing real RETRACE parentage, goal switches, protected-cell invariants, and terminal-goal success. This experiment is a bounded throughput/curriculum probe, not an argument to build more dispatch infrastructure.

DRIV3R PR #1 stays Draft/HOLD, unmerged, and its main remains unchanged.
