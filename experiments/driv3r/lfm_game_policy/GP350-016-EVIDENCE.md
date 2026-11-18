# GP350-016 — Original GP350-015 full-model evidence audit (2026-11-16)

**Disposition:** No intermediate GP350-014 checkpoint qualifies for promotion. This is an independent read-only analysis of previously completed LFM2.5-350M inference, **not** a new full-model training or closed-loop evaluation.

## Custody and execution

- Historical (read-only) full-model run: [CyberSand Actions #43943992509](https://github.com/techrote/cybersand/actions/runs/43943992509), completed **2026-11-16 20:30 UTC**, SUCCESS.
- Original immutable artifact: [#14266902004](https://github.com/techrote/cybersand/actions/runs/43943992509/artifacts/14266902004), ZIP digest `sha256:a4a83e6b1bfcbdb05e35b96e3adc9b218dafe59b0c6b76c416c51efd0162469f`.
- Exact original experiment source: `a09470a10f4dfc7cbdb83753ea3d8eded533a444`; original model revision `9921cd4d82c0e3986887aa4b376f9446729ef8a3`.
- Independent byte/receipt audit: 87 archive members with valid CRC, 36/36 original W0 PNG SHA-256 hashes, **2,304/2,304 full-model decision score vectors**, legal-choice and host-application agreement, 360 archived held-out probes, and checkpoint digests against `gp350-015-request.json`. **0 illegal host applications**.
- Reproduce the derived audit using `python3 experiments/driv3r/lfm_game_policy/gp350016_receipt_audit.py gp350-015-original.zip` after downloading the exact ZIP. No network/model weights/dependencies are required.

## Frozen checkpoint comparison

| Original GP014 adapter | Held-out general /72 | Trapped RETRACE /24 | False RETRACE /24 open probes | On-policy trapped RETRACE | Premature RETRACE | BRANCH | Mean unique cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| step18 | 36 | 9 | 0 | 0/0 encountered | 0 | 0 | 65.00 |
| step36 | 35 | 9 | 0 | **10/12 encountered** | **45** | 76 | **56.58** |
| step54 | **43** | 4 | 0 | 0/0 encountered | 0 | 0 | 65.00 |

The GP006 conservative parent was previously 41/72 general with zero autonomous RETRACE; GP014 step72 reached 13/24 trapped-probe RETRACE but produced 25 premature autonomous RETRACEs. These historical controls were **not** independently rescored in GP350-015.

**Step36 is not a recovery winner.** Its 10 apparent successful retraces are entangled with 45 premature ones, 76 branches, 38 raw illegal preferences and 8.42 fewer distinct cells per episode than step18/54.

## Critical newly identified effective-sample-size defect

The 768 decision receipts for each checkpoint do **not** represent 768 distinct model observations. Deduplicating the complete canonical observation (including `x`, `y`, tick, heading, 5×5 context, legal menu, motif and action history) yields:

| Checkpoint | Unique observations | Repeated exactly across evaluation seeds | Distinct trapped observations |
|---|---:|---:|---:|
| step18 | 655 | 113 | 0 |
| step36 | **502** | **266** | **6** |
| step54 | 683 | 85 | 0 |

For step36, **all six trapped states in seed 9521 are byte-identical to the six in seed 9503**, with identical six-way scores. Thus its 10/12 successful autonomous trapped decisions represent only **five unique successful decisions on six distinct traps**, not twelve independent opportunities. Nominally different evaluation seeds do not imply held-out state novelty. This is an evidence-strength limitation, not a model runtime bug.

## Model-side logit sensitivity: observational only

On the 768 already-recorded step36 observations, subtracting a constant **0.5** from the RETRACE logit would reduce premature choices **45 → 2** while retaining **10/12** trapped retrace choices. Penalty **1.0** yields **0** premature choices but only **6/12** trapped RETRACEs. Such results are **off-policy score re-rankings**: changed actions alter later states, so neither trajectory coverage nor autonomous recovery improvement is established.

The observed separation motivates a calibrated inference-side ablation, but using the two recorded seeds as train/test would leak all six positive trapped states. First construct and freeze a more diverse, disjoint trapped-state curriculum; retain original GP006, untuned stock, d1 when matched, and GP014 checkpoint controls; then conduct genuine fresh LFM rollouts and bounded optimizer comparisons.

**Boundary:** synthetic W0 only, no native CyberSand evidence; no new LFM forwards or optimizer updates in this audit, no model/checkpoint promotion, no outside-repository writes, no paid compute. Remain Draft/HOLD, do not merge.
