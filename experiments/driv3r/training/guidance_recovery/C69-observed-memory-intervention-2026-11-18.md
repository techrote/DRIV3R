# C69 — Observed-route memory intervention potency (18 November 2026)

**Disposition: DRAFT/HOLD, do not promote.** Synthetic-host evaluation only; no new optimizer run, production ABI inference or native CyberSand acceptance. Ownership: `techrote/DRIV3R` PR #1, directory `experiments/driv3r/training/guidance_recovery/`.

## Predeclared design and exclusions

C68 showed 140/144 identical C60-FROZEN movement+painting traces across 0% versus 50% erasure in remote-goal cohorts. The next prerequisite is *observable intervention potency*, not additional hint-selector fitting. C69 specifies seed `690118`, five graph families (`tree`, `braid`, `prim`, `wilson`, `kruskal`), four graphs per family, prefix ticks 19/29, host-observed frontier target, at least six observed menu positions and 19 executed prefix actions, at least three observed route edges, and at least three recorded edges removed under full erasure. The full-erasure route must disappear **before any suffix rollout**. Select the first eligible prefix of each graph and the first four eligible graphs per family in deterministic generation order. Exclude known historical graph hashes from original C60/C61/C64/C65/C66/C67 receipts. Goals and policy outcomes must not enter eligibility filtering.

All 20 selected graphs have different SHA-256 graph identities. Exactly four were selected in each of the five families. Prefix observation routes were 4–8 steps, with 6–18 known menu positions. Under full erasure, the retained contemporary menu at the current position had only two recorded directed edges; every selected previously available route to the observed waypoint disappeared (20/20). At 50% erasure, 19/20 selected routes disappeared. These data define a **purposely memory-rich, observed-waypoint cohort**; they cannot be interpreted as randomly sampled remote-goal challenges.

Controls: fixed frozen C60 neural weights, non-neural C60-FRONTIER, C34-RESET and C18-A0; no cue, weak backtrack (`0.35`) and strong backtrack (`1.0`); intact, 50% and 100% erased historical memory. The host's fresh legal action menu and protected-cell checks are authoritative. Intervention removes previously observed menu/action history; no alternate graph oracle, direct remote map access or illegal-action injection. Each graph is paired under the same initial simulator and PRNG state. Local instrumentation counts actual `frontiers()` and `route()` calls on the observed ledger without modifying the historical source module or the game.

## Executed results and key finding

720 distinct synthetic-host policy suffixes (20 graphs × 3 cues × 3 memory arms × 4 policies) executed 14,539 host-legal actions. All host rejections were zero, protected-cell hashes preserved, and all no-guidance movement+painting hashes remained identical across policies and memory arms. Two independent CPU executions produced byte-identical JSON summaries, complete ledgers and SVG gallery. Six evidence-verification tests passed.

Paired changes versus intact memory, out of 20 graph identities per row:

| Policy and cue | 50% memory: changed traces / goal delta | 100% memory: changed traces / goal delta |
| --- | ---: | ---: |
| C18-A0, no guidance | 0/20; 0 | 0/20; 0 |
| C34-RESET, weak or strong | 0/20; 0 | 0/20; 0 |
| C60-FRONTIER, weak | **10/20; −1** | **6/20; +1** |
| C60-FRONTIER, strong | **13/20; +2** | **10/20; +3** |
| C60-FROZEN neural, weak | **1/20; 0** | **2/20; 0** |
| C60-FROZEN neural, strong | **1/20; 0** | **4/20; −1** |

Strong guidance with intact memory completed 17/20 observed-waypoint goals under C60-FROZEN; full erasure reduced this to 16/20. The non-neural frontier policy completed 12/20 strong-guidance goals intact and 15/20 after full erasure; the apparent gain is not a memory-resilience advantage and may reflect escape from previously misleading historical routes. Paired goal effects in this small outcome-selected-by-memory-richness cohort are exploratory. A graph-resampling 95% interval for strong-cue full-erasure minus intact goal counts is [0,+7] for FRONTIER and [−3,0] for FROZEN, not a confirmatory test.

The pre-intervention route was explicitly removed on 20/20 graphs. Nevertheless, the frozen neural controller rarely changed behavior, whereas an explicit memory-reading frontier controller changed on up to 13/20 paired episodes. The resulting **mechanistic interpretation is narrower and stronger than C68's**: this time the deletion demonstrably changes accessible route memory, and runtime route reads are instrumented; the frozen neural policy is relatively *insensitive* to that historical information in this cohort. This does **not** show robust neural recovery from consequential memory damage. Its `frontiers()` feature does read the ledger but it does not directly call `route()`; zero `route()` calls for the neural controller are expected, not missing telemetry. `route()` accesses for the FRONTIER policy are genuine but alone cannot prove causal contribution to a given action, and a changed early action can propagate stochastically throughout the suffix.

## Scientific limitations, custody and next gate

The complete source and regression verifier are committed in the authorised laboratory. A compact 20-graph paired-result index is also committed, along with the SHA-256 of the **full** executed ledger, SVG gallery and summary. The full 63,753-byte encoded episode ledger, 14,452-byte trajectory gallery and 64,823-byte summary are retained in the separately hash-verified owner-evaluation archive; **they are not yet in Git or immutable GitHub Actions artifact custody**. The PR's current source and index are not an independent complete research record until that gap is closed.

Reproduce on free/local CPU only by unpacking the owner archive and running:

```sh
export C69_ARCHIVE_ROOT="$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -W error -m unittest -v test_c69_memory_potency.py   # after setting C69_EXECUTED_DIR to existing executed/ directory
python c69_memory_potency.py /tmp/C69-independent
C69_EXECUTED_DIR=/tmp/C69-independent python -W error -m unittest -v test_c69_memory_potency.py
```

Verified environment: Python 3, NumPy 2.3.5, PyTorch 2.10.0+cpu, safetensors 0.7.0. Frozen predecessor source is the unchanged C64 owner-archive snapshot; upstream C60 module Git blob `b7ceed0ee3c46b4947628766be3ad33a52023620`. C69 executable source Git blob `47013630b05da7495c404945973948c833029264`, SHA-256 `5ab636ce0833d671f9e9027e07b48d98e98382f30f861a582dd629b22603f849`. Full ledger SHA-256 `c052e5da9d03b035e32196c49fd611f38fff4f9c61bcc9599e716bd404b93486`. This does not assert bitwise checkpoint reproducibility, because no model was trained.

**Next:** avoid training on this selected observed-route cohort as though it proved remote-goal robustness. Separately specify an *unseen remote-goal* cohort with multiple consequential historical branch opportunities and a genuine consumer of those memories; retain a balanced never-seen independent test. Improve action-policy recovery only after host-observed intervention potency and full personality measures are established. Keep all future work confined to DRIV3R PR #1, Draft/HOLD.
