# Wave 19 — live loop-guard experiment, research source qualification

16 November 2026 · **DRIV3R Draft/HOLD PR #1** · **source-qualified, not model-qualified**

## Concrete migration blocker discovered

The imported `experiments/driv3r/frankenlayers/infer.py` defines
`run_arm(binary, model, w0, mode, out)` and reads `SEED` from module globals.
The historical lab `counterfactual.run_variant()` instead calls
`paired.run_arm(binary, model, host, mode, out, seed)` (six arguments).
Its historical observation/seed contracts also rely on globals. These sources
can compile successfully but are **not runtime-compatible as imported**.
The previous source-only import qualification must not be mistaken for scientific
reproduction. Existing historical source and evidence are preserved unchanged.

`wave19_live.py` introduces an **additive lab-owned compatibility path**:
it binds the imported five-argument parent API and every seed/step/family
global in a reversible fold context, then restores them on success or exception.
It does not change `frankenlayers/infer.py`, `counterfactual.py`, the immutable
`d_common` fixture, or CyberSand. The recorded action proposal, raw probabilities,
trained residual choice, shield decision and loop-escape rationale are retained
separately for source auditing.

## Bounded manual real-model experiment prepared

The [request](run-request-wave19.json) is immutable and opt-in only.
AERO4_NO2 `(0,4,10,20)` Q4_K_M is compared with the same-runner Q8 stock,
Q4 stock, raw four-layer carcass, trained strong residual, original wall
shield and Wave-18 loop-aware shield. A single 55-parameter CPU residual is
trained using **at most 96 actual frozen-model decisions** from seed
`2718/architecture`, plus explicitly labelled synthetic STOP augmentation.
All model weights remain frozen.

Two *previously unused* evaluation groups, `6089/pipework` and
`7127/greebles`, shift geometry independently. Four matched variants per
group (forward gate and north/south side obstacle), 24 decisions per variant,
six arms = **1,152 maximum held-out inference decisions**, 1,248 including
training. No far-gate arm is wasted on an often-unexposed intervention.
Both pairs are scored with **exposed-pair denominators**, with novelty,
revisits, premature STOPs, host legality, action entropy and same-runner
decision latency retained as independent axes. A changed action alone is
not credited as obstacle *recovery* or goal completion.

Manual source-pinned invocation, only in a temporary workspace with known
runtime/model assets:

```bash
python -m unittest experiments.driv3r.frankenlayers_lab.test_wave19_live
python -m experiments.driv3r.frankenlayers_lab.wave19_live \
  --binary /path/to/pinned/llama-server \
  --model-q8 /path/to/pinned/d1-2b-q8.gguf \
  --model-q4 /path/to/verified/derived-q4.gguf \
  --quant-receipt /path/to/q4-quantization-receipt.json \
  --w0 /path/to/source-pinned/w0.py \
  --source-sha <checked-out-DRIV3R-commit-sha> \
  --output /path/to/new-immutable-evidence-directory
```

Keep full process logs, failure artifacts, real HTTP responses, action traces,
checkpoint hashes, W0/Q8/Q4 SHA-256 receipts and input/output hashes as
immutable GitHub Actions artifacts *if* a bounded explicit manual runner is
supplied. Do not import large checkpoints into Git. Do not schedule training
on PR or push and do not create additional experiment-specific workflows.

## Qualification and limits

* New source tests exercise parent five-argument call binding, fold-global
  restoration on exception, per-episode loop-state reset, raw/learned/host
  action legality, exact runner ABI, clean checked-out SHA and multi-axis scoring.
* Six isolated source tests executed locally with dependency stubs:
  **6/6 passed**. This checks the new control logic, not integration with
  actual models or the imported dependency tree.
* The repository's source-smoke [run 43979259782](https://github.com/techrote/DRIV3R/actions/runs/43979259782)
  passed on the request commit; it checks Python syntax and forbidden
  automatic workflows, **not** the four unit tests or real inference.
* Neither the two held-out evaluation folds nor model/adapter training has
  been run as part of this contribution. No new performance or recovery
  outcome is claimed. The current historical leader remains the Wave-17
  strong residual; the Wave-18 loop guard has only first-divergence
  shadow evidence.

**Boundaries:** Only files under `experiments/driv3r/frankenlayers_lab/**`
were written to the existing DRIV3R research branch. CyberSand and Neurounner
were read-only; no game implementation, production settings, paid compute,
new PR, merge or Ready promotion.
