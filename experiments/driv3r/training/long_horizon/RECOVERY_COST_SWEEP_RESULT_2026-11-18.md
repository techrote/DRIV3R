# DRIV3R — RECOVERY-COST-SWEEP-001 (18 November 2026)

**Finding: bounded positive on sampled autonomous goal arrival, with a safety tradeoff; not general recovery competence.** A fourfold training-sampling emphasis on feasible BACKTRACK examples raised sampled reachable-target arrival from **106/128 to 115/128**, while producing **12 futile initial retreats** versus zero in the matched standard-supervision control. Twofold emphasis gave no arrival gain. A deterministic exact-stack rule still dominates this specifically constructed task, with 100% first-action classification versus the best neural policy's 95.2%.

## Provenance and ownership

- Only `techrote/DRIV3R`, Draft/HOLD PR #1, branch `research/import-consolidated-worm-labs`. Source head before editing: **`df0ae48772e62ae302b515b57937c629f6c0292f`**. No changes to DRIV3R main, CyberSand or Neurounner.
- Only files in `experiments/driv3r/training/long_horizon/`. The frozen synthetic World, menus, older neural code and historical evidence are **not modified**. Verified original core-module SHA-256 hashes against CAP-CONDITIONING-001 source ledger before running.
- TRAIN and DEVELOPMENT have disjoint motifs/spec seeds; 336 synthetic TRAIN state cases versus 480 DEVELOPMENT cases. The structural HELDOUT split remains sealed.
- Existing source-faithful synthetic fixture only. Changing `max_recoveries` at the fork is an **experimental counterfactual**, not an established CyberSand host ABI operation.

## Experimental design

The target is a previously traversed, currently painted position. The task-specific teacher chooses **BACKTRACK iff the exact ordered history places the target within remaining recovery credits; otherwise STOP**. This is not a general-purpose gameplay action menu policy. All original alternative legal menu actions remain offered in the evaluation, including the synthetic `RELOCATE_START` operation; their frequency is audited.

Three otherwise identical 24-hidden-unit neural rankers use the same parameter initialization per seed, equal optimizer updates, unchanged host-derived observations and identical original training cases. Only the relative sampling frequency of positively labelled BACKTRACK examples changes: **1×, 2×, 4×**. This is class-cost-sensitive supervised imitation, *not* end-to-end reinforcement learning. Independent optimizer seeds: 941 and 942. Each runs 96 optimizer steps at each nominal stage 36, 72 and 128 (288 per checkpoint). Training credit probes are 2/4/6; evaluation credit probes 1/3/5. No large pretrained model and no paid API/GPU were used.

We censused all **8,640** DEVELOPMENT probability/action vectors (480 cases × 3 credits × 2 seeds × 3 arms). For trajectory outcome, a **predeclared systematic sample** of 48 case states (both alias kinds, 20 motifs represented, six sampled rotation/original-budget combinations), independently evaluated under each model/credit condition, generated **864 actual synthetic continuations** of up to seven host-legal actions. That is **288 continuations per arm**, including 128 analytically recoverable and 160 unreachable in each arm. All actions were applied to mutable copied Worlds with cap-specific offered menus. `CapMenu` is a read-only feature-presenting adapter; it does not mutate World or bypass legal action checks. No simulated action was counted without an accepted receipt.

## Results

| Supervision | First decisions correct / 2,880 | Missed recoverable first decisions | Futile initial BACKTRACK | Reached recoverable target / 128 | Recovery actions (sampled) |
|---|---:|---:|---:|---:|---:|
| Standard 1× | 2,639 (91.6%) | 241 | **0** | 106 (82.8%) | 156 |
| Recover 2× | 2,640 (91.7%) | 240 | **0** | 105 (82.0%) | 157 |
| **Recover 4×** | **2,742 (95.2%)** | **126** | 12 | **115 (89.8%)** | 171 |
| Exact-stack deterministic teacher (classification reference) | **2,880 (100%)** | 0 | 0 | Not independently rolled out | — |

The fourfold training arm gained **9 additional arrivals among 128 recoverable sampled continuations**, or +7.0 percentage points on that conditioned denominator. A motif-clustered descriptive bootstrap on all 288 continuations/arm gives +3.33 percentage points unconditionally (95% interval +1.04 to +5.83). This interval is **exploratory, unadjusted for two alternative class-emphasis arms, and based on correlated, systematically selected synthetic worlds**. Do not interpret it as evidence for deployment generalization.

The improvement appeared in both training seeds but was uneven: seed 941 gained 7 arrivals among 64 recoverable; seed 942 gained 2 among 64. The fourfold arm introduced 8 and 4 futile *initial* retreats in the respective seeds. The sampled unreachable cases all stopped without spending recovery credits, but that smaller sample did not capture the 12 false retreats observed in the complete first-action census. Do not infer that the false-retreat risk disappeared downstream.

None of the sampled continuations selected `RELOCATE_START`. Its legal availability was preserved; the zero frequency is an observed result, not an artificial filter. A full six-panel SVG compares complete synthetic cursor histories, current history stacks and executed continuations. Gallery cases are illustrative and not a substitute for the scored census.

## Qualification

- 4 focused tests: **3 source-only PASS**, one optional full-evidence test **PASS** when run with `DRIV3R_COST_RUN`.
- Recreated all 8,640 model probability vectors from saved checkpoint tensors, rescored all 864 selected continuations, checked case summaries and cluster bootstrap, and independently replayed **114 recorded host-legal action receipts** in 108 sampled full transcripts, restoring the complete starting prehistories.
- 6 real trained checkpoints, 24-unit hidden rankers, 288 optimizer updates each, SHA-256 recorded in the adjacent evidence ledger. Checkpoints are **not committed into Git**.
- Measured ~14.7 seconds total CPU run wall time (environment dependent), ~173 MiB peak process RSS, approximately 0.010 ms median model-only probability inference. See machine ledger for exact values. Entire run is CPU-only, bounded, and opt-in.
- Complete raw model probabilities, checkpoint tensors, receipts, full trajectories and source are retained in a separately hashed local evidence ZIP. They have **not** been published as immutable GitHub Actions artifacts; a local archive is not a substitute for immutable upstream evidence custody.

## Decision and next work

**Retain the 4× model as an exploratory checkpoint, not an accepted policy.** It reduces missed reachable backtracks but incurs some unnecessary retreat and remains inferior to a simple analytic stack rule. Do not train larger d1/LFM/Sprocket adapters on this evidence alone. Next test: cost-sensitive classification with nonzero penalties for futile retreat plus truly downstream 36/72/128 *executed* actions, new disjoint dynamic goals and material/obstacle constraints, comparing deterministic stack-augmented control and neural policies with fixed source-matched native action receipts. Structural HELDOUT and native CyberSand acceptance remain outstanding.
