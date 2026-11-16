# GP350-017 — unique-state RETRACE margin calibration (16 Nov 2026)

**Disposition: negative for a learned scalar fix. No LFM checkpoint promotion.** This independently analyzes original GP350-015 full-LFM inference and adds **320 actual AdamW updates to two one-parameter post-hoc scalar calibrators**, not new LFM/LoRA training or forwards. Counterfactual choices are evaluated on archived observations only, **not** closed-loop gameplay trajectories.

## Source custody

- Immutable read-only historical run: [CyberSand #43943992509](https://github.com/techrote/cybersand/actions/runs/43943992509), [artifact #14266902004](https://github.com/techrote/cybersand/actions/runs/43943992509/artifacts/14266902004).
- Original source commit: a09470a10f4dfc7cbdb83753ea3d8eded533a444; base model LiquidAI/LFM2.5-350M at revision 9921cd4d82c0e3986887aa4b376f9446729ef8a3.
- Original ZIP SHA-256: a4a83e6b1bfcbdb05e35b96e3adc9b218dafe59b0c6b76c416c51efd0162469f.
- Original receipt audit: **87 ZIP entries**, **36/36 archived PNG hashes**, **2,304/2,304 scored model decisions**, host-legal action agreement, three checkpoint hashes, and the 360 original held-out probes accounted for.
- Step36 checkpoint SHA-256: e37999047767861cfb7c05d64310d503505eeff3795c755d547b5e8aaa81cdf7. Checkpoint binaries are not in Git.

## Why the world-size split matters

The original experiment uses two nominal seeds, but these share numerous exact observations. Deduplication by **complete canonical observation** yields:

| Frozen GP014 adapter | Unique observed states | States shared between sizes | Unique locally trapped states |
|---|---:|---:|---:|
| Step18 | 655 | 56 | 0 |
| Step36 | 502 | **0** | **6** |
| Step54 | 683 | 28 | 0 |

Step36's six unique traps split three each between 32-cell and 48-cell worlds. The **32 ↔ 48** comparison is exact-observation-disjoint; nominally different seeds are not counted as independent replicates. Each held-out test contains only **three trapped states**, imposing a severe effective-sample-size limitation. For step18/54, no trapped state arose on-policy: lack of premature RETRACE does not establish useful recovery.

## Genuine tiny optimizer experiment

For step36 only, fit a single additive score bias b in logit(RETRACE) + b against the synthetic W0 locally-trapped predicate. Use 160 CPU AdamW updates per direction (320 total), learning rate 0.05, class-balanced BCE and 0.005 × b² regularization. The actual six-way LFM scores and original observations are frozen; neither base model nor LoRA receives gradients.

**TP** means RETRACE when the local synthetic definition says trapped; **FP** means RETRACE while an unvisited move remains legal.

| Held-out world size | Original step36 | Learned scalar offset | Fixed −0.5 offset | Geometry-label comparator |
|---|---|---|---|---|
| 48 (292 distinct, 3 traps) | TP 2, FP 23 | **TP 2, FP 25** (b=+0.1660) | TP 2, FP 0 | TP 3, FP 0 |
| 32 (210 distinct, 3 traps) | TP 3, FP 8 | **TP 3, FP 28** (b=+0.7240) | TP 3, FP 1 | TP 3, FP 0 |

**Learned calibration worsened premature retracing in both held-out world sizes.** Fixed −0.5 gives better retrospective discrimination, but is an **untrained sensitivity control** and already explored against related historical observations; it is not independent evidence of a promoted policy. The geometry-label comparator achieves all six true positives and no false positives by inspecting the very predicate used to define the label. It is a tautological transparent comparator, **not a learned accomplishment**.

Actions re-ranked after the fact would change later observations if actually applied. There are **zero new intervention rollouts**, zero demonstrated coverage gains, zero full-LFM forwards, zero full-LFM optimizer updates and zero native CyberSand gameplay outcomes. No neural recovery/generalization or causal advantage has been established.

## Reproduction and artifact custody

The [source-only auditor and scalar optimizer](gp350017_unique_state_calibration.py) and [complete machine-readable measurements](GP350-017-MEASUREMENTS.json) are preserved exclusively inside the DRIV3R LFM research directory.

    python3 experiments/driv3r/lfm_game_policy/gp350017_unique_state_calibration.py --self-test
    python3 experiments/driv3r/lfm_game_policy/gp350017_unique_state_calibration.py gp350-015-original.zip --out gp350017-new-evidence.json

The ZIP must match the pinned original digest. Source checks, both actual scalar optimizer fits, PNG/receipt audits and output generation passed locally. Python 3 and CPU PyTorch are required for fitting; no paid accelerator, network call or new Actions workflow. The result path must not already exist.

**Next evidence gate:** predeclare novel, disjoint trap topologies and split by world and exact observation; then run fresh full-model inference and actual host-applied closed-loop trajectories, with GP006, GP014, untouched stock and eventually matched d1 controls. Measure legality, recovery usefulness, coverage, novelty, latency and RSS. Do not modify the CyberSand game or its canonical host.

**Ownership:** techrote/DRIV3R PR #1 only, Draft/HOLD and unmerged. CyberSand and Neurounner were read-only references. No production settings or native assets changed.
