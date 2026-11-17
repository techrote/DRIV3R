# DRIV3R Corpus Factory v20 — multi-receipt topology transfer

17 November 2026. Synthetic-host **DATA ONLY** in `techrote/DRIV3R` Draft/HOLD PR #1. No native CyberSand acceptance or SLM training claim.

## Motivation and novelty

V19's repair corpus reused v17/v18's narrow two-lane ordered-instruction fixture, with a one-purpose symbolic finite-state solution. V20 is an independent synthetic two-dimensional grid host: distinct TRAIN/DEV/HELDOUT spatial ranges, procedurally generated protected-cell barriers, detour routes, blind spurs and dynamic dual-shutter closure after eight accepted movement steps. Every action after a shared historical prelude is executed against the replay host; accepted paints and protected refusals are actual **synthetic-host** receipts. The host does not represent CyberSand's native action/observation ABI.

The authorized workshop branch is determined by XOR of **three spatially remote independent receipt bits**. Each world group contains all eight bit combinations. They share a 64-event/position suffix, present-local 5×5 view, legal action menu and objective, but four require NORTH and four SOUTH as their first expert move. The complete three-bit history is necessary for this constructed first-action classification; any one or two historical bits, as well as the last64 suffix alone, have a **50% ceiling** in each balanced group. This is a mathematical benchmark property, not evidence that a trained neural policy reasons over three receipts.

From a hub, a policy must navigate the selected workshop under partial 5×5 observations, respond to the two shutters closing, legally paint five cells in an alternating SAND/METAL/GLASS/ASH motif and return home to STOP. The other workshop must remain unpainted. Some episodes begin with an intentionally refused protected-cell probe and matching receipt. Actions and history after the prelude are model-controllable via `createHost`, `observe` and `apply`; the oracle supplies training labels and validates completed trajectories only. This is a new synthetic topology implementation rather than a renamed v17/v18 dependency.

## Partitions and provenance

| Split | Disjoint topology groups | Episodes | Expert action labels | Branch length | Detours per branch |
|---|---:|---:|---:|---:|---:|
| TRAIN | 20 | 160 | 6,872 | 14–19 | ≤2 |
| DEV | 8 | 64 | 3,712 | 20–25 | ≤3 |
| HELDOUT | 8 | 64 | 5,056 | 26–33 | ≤4 |
| **Total** | **36** | **288** | **15,640** | — | — |

A deterministic `v20:<split>:<group>` seed controls the world geometry. All 36 world geometry SHA-256 fingerprints are distinct, with no cross-split world group reuse. This is **not** a perceptual near-duplicate guarantee: two worlds may still share the same synthetic corridor grammar. Episode variants remain together within their world split. Each episode's source `(split,group,variant)` is recoverable from its identifier. `initial_model_input` and each decisions shard's `model_input` are the *only* model-visible feature sets. Labels, receipt outcomes, evaluation metadata and `makeRecord`'s `private_target` must never be fed to the model at inference.

## Qualification and reproducibility

- `node test_topology_relay_v20.js`: **7/7 PASS**. Tests cover unique split topologies, 1-/2-receipt and last64 aliasing, held-out closed-loop expert rollouts, protected-cell rejection, timed dual shutter closures, model-visible/privileged separation, two byte-identical complete six-shard generations, SHA-256 validation and refusal to overwrite an existing output directory.
- `node topology_relay_v20.js NEW_EMPTY_DIRECTORY 1`: **288/288 expert episodes reach a correct terminal**; 15,640 accepted expert actions (including 288 STOP actions), 288 shutter events, 104 intentionally refused preflight protected moves, zero subsequent oracle-refused actions and zero accepted protected-cell entries.
- Local generation on Node.js 22.16.0 Linux took **3.74 seconds**, peak RSS approximately **79 MiB**, including six-shard SHA hashing. These numbers are generator metrics, **not SLM inference latency/RSS**.
- Manifest SHA-256: `7228052cbd6b7f3322659751a66b4570ad6d00a7771d259ad3ab16d0141e903f`. Source Git blob: `dc7188c6b97a9550b6d406049bdfe20a5368afa9`. Test Git blob: `0fe7e24dc557ff7d1ff08b6e3b618cdd40a471cf`. The six full 172,721,150-byte uncompressed JSONL shards are regenerable; they are deliberately not committed to Git. SHA-256 for every shard is retained in `V20_DEFAULT_MANIFEST.json`.

## Scientific failures / honest bounds

1. Although three old receipts are independently necessary for the first-action label under the controlled projections, XOR remains a **small deterministic symbolic rule**. This dataset cannot prove complex learned memory, generalization or competence in CyberSand.
2. All worlds retain a hub/two-workshop skeleton; variations in length, detour, protected cells and side branches do not establish broad cross-topology diversity. There is no independently audited perceptual near-duplicate threshold.
3. A deterministic oracle controls the labelled trajectories; this is **not** trained SLM control, and no CPU/GPU model weights were changed. After-action decisions and shutter receipts are executable but have not been exercised by a learned model.
4. The initial 80-step spatial history and three remote receipts are scripted. The painting, shutter and protected-refusal semantics are synthetic; not native CyberSand receipts. No immutable Actions artifact has been published, and no host-legal CyberSand ABI qualification has occurred.

## Consumer contract and next experiment

```sh
node experiments/driv3r/corpus_factory/test_topology_relay_v20.js
node experiments/driv3r/corpus_factory/topology_relay_v20.js /new/empty/directory 1
```

For each episode, reconstruct from `makeRecord(split,group,variant)`, then call `createHost(record)`, `observe(host)` and `apply(host,model_action)` at **every step**. Use TRAIN only to fit a real pinned LFM/Sprocket SLM, DEV for selection, and finally one sealed HELDOUT test. Compare full history vs last64 vs two old receipts vs zero history. Report world-grouped first-action accuracy, 5-cell exact paint success and inactive workshop cleanliness, stopped/completed episodes, protected rejection counts, extra actions/path stretch, p50/p95 model action latency and peak RSS. Preserve tokenizer and model hashes; checkpoints belong in immutable Actions artifacts, not Git.

PR #1 stays Draft/HOLD. No game, Neurounner, main-branch or other research-lane writes.
