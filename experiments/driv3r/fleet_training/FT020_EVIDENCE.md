# FT-020 — four-policy inference residency and throughput (16 Nov 2026)

**Disposition:** local CPU research probe **completed; hosted Actions qualification pending**. This experiment belongs solely to the existing DRIV3R Draft/HOLD PR #1 and is not FT-019 checkpoint inference or native CyberSand acceptance.

## Research question and custody

With four distinct, genuinely Adam-trained 24-hidden-unit NumPy policies, compare four processes holding one model each against two processes holding two each and one process holding four. All modes use equal fixed checkpoints and identical synthetic inference menus, independent lease IDs, process-level RSS limits and preserved per-worker stdout/stderr logs.

- Executor: **ft020_resident_inference.py**; SHA-256 **d5b8e261004af20095687b764c729fe5535a2a6a6be9e17a6152c6a24f4f4fc1**; Git blob **3fe3e8d2438b0dd1c24dbfe218f9b8200c8bb767**.
- Tests: **test_ft020_resident_inference.py**; Git blob **b4627fa84fe52f4ce4036789b2f7c9b256b7b9a6**; **3/3 PASS** locally.
- Historical population source reference: experiments/driv3r/training/population/population_lab.py; imported Git blob **8a974a072dd8a5172f9242d83a373f3de345bb3b**. FT-020 uses the same 20 named feature slots and 20×24×1 inference shape, but its **independently generated one-step menus and newly trained weights are NOT a replay of that host implementation**.
- Runtime: Linux container, NumPy 2.3.5, 5 visible CPU affinity cores, BLAS threads forced to one. **Not a GitHub-hosted Actions runner**; relative performance may change on different hardware.
- Training: four unique niche leases (navigator, recoverer, branch_artist, material_cautious); 180 real NumPy Adam updates each, 128 training menus plus 96 held-out menus per niche. Total **720 Adam updates**; neither a pretrained SLM nor a full sequential game episode.
- Budget: 180 seconds total, 5,000 inference menus/model maximum; 450,000 KiB per process and 900,000 KiB sum-of-sampled process RSS caps. No network, paid compute or game-side changes.
- Model checkpoints and binaries have **not** been committed to Git. Original local checkpoints, reports, worker logs and SHA256SUMS exist in session-local evidence but **not immutable Actions artifacts**.

## Matched measured throughput

All three trials passed exact per-policy action-choice parity across all three residency modes, with identical checkpoint hashes. Four separate inference processes and two separate processes both had observed overlapping advancing CPU-tick counters in their respective phases.

| Synthetic menus per model | 4 processes × 1 model | 2 processes × 2 models | 1 process × 4 models |
|---|---:|---:|---:|
| 800 (3,200 decisions) | 2,176 decisions/s; 1.471s | **2,952/s; 1.084s** | 2,688/s; 1.190s |
| 1,200 (4,800 decisions) | **4,151/s; 1.156s** | 4,037/s; 1.189s | 3,493/s; 1.374s |
| 5,000 (20,000 decisions) | **12,944/s; 1.545s** | 9,389/s; 2.130s | 5,628/s; 3.554s |

Whole-phase wall times include spawning the child, Python/NumPy imports and all inference; they are not pure matrix-kernel latency. The host's CPU entitlement, caching and process scheduling were not independently controlled. For the smallest workload, two models/process saved startup cost; for the largest workload, four separate inference processes significantly outperformed model sharing. **Do not declare a universal two-resident winner.**

| Residency | Sum of separately sampled per-worker RSS peaks |
|---|---:|
| Four processes × one | ~391,100 KiB |
| Two processes × two | ~195,550 KiB |
| One process × four | ~97,770 KiB |

These are sums of process-specific peaks, **not exact concurrent peak RSS**. Python/NumPy interpreter overhead dominates the tiny weight tensors.

## Checkpoint identities and held-out negative results

| Independent niche | Checkpoint SHA-256 | Held-out top-action accuracy before → after | Held-out cross-entropy before → after |
|---|---|---:|---:|
| navigator | 1debec1ad87fd49a6e8304d875ab766fa0e72ea3dca28ab68846beff12f2b5af | 32.3% → 71.9% | 1.489 → 0.660 |
| recoverer | 22b99c3b807e2c161d9e47acb6e317658c4db41a783eb97aa5c7b92d82d44310 | 21.9% → 62.5% | 1.572 → 1.072 |
| branch_artist | 5493d1cbc0df1c18da10bb0b1ab6519e42d383e120ff78a0a9ddd3645d8e3fc7 | **45.8% → 43.8% regression** | 1.372 → 1.069 |
| material_cautious | 4a763d88fced6831e1327ae1227e983e4bb6d61f9c00656f2e1d1a5c95215cac | 31.3% → 72.9% | 1.574 → 0.713 |

The branch_artist accuracy regression is preserved; its cross-entropy improved but its top-choice accuracy did not. These synthetic teacher-score metrics are **not host-authoritative CyberSand gameplay metrics**.

## Original local evidence checksums

- 800 menus/model: report.json SHA-256 **e9418dd5a36b5e5c91d24c94be885001ba284ab229577dd8e7f019979c2ead2a**; SHA256SUMS SHA-256 **187c74972966770b5bd3d6a8b9045fa0c78ef7dcb7f357a49b2814f426117403**.
- 1,200 menus/model: report **fa4579d9a5c9fafc63bae4e72ee5f1f86bec080de5494f90a8a0ad0d293e1f63**; manifest **8350a404b2339b63ab7024716961f6f06a0bbdc166bb82cf09fc4ad40cb8bca9**.
- 5,000 menus/model: report **c355c87cbc55aa19a51504290d8ff7ec6e2387fce2ff875a3852eccd38ad482e**; manifest **4b1f7001fcde942757c76359f249c263e6af00a9528a1f322c32e6ca3e203515**.
- No failed workers in these three trials; failures, when present, retain logs and a failed-run report where possible.

## Follow-on qualification

Manually rerun the exact source on a single **GitHub-hosted CPU runner** under an approved, bounded, opt-in common research mechanism, and retain its original ZIP in Actions artifacts. Do not create another push-triggered experiment-specific workflow. Repeat workloads to measure variance and CPU quota saturation. Benchmark **actual FT-019 saved checkpoint** inference and synthetic-host legal action trajectories only once their original archived models are recovered and verified. The present data support provisional runner sizing, not claims of policy intelligence, full-game behaviour or scientific reproducibility of the historical imported source.

No CyberSand/Neurounner writes. DRIV3R PR #1 stays Draft/HOLD and unmerged.
