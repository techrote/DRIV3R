# DRIV3R Corpus Factory v42 — delayed dual-interlock causal memory

18 November 2026. Synthetic host only; not native CyberSand. This is a corpus-generation result, not a trained model or measured model inference. The sole writable location is `experiments/driv3r/corpus_factory/**` on existing DRIV3R Draft/HOLD PR #1.

## Research question and design

V41's dynamic-obstruction benchmark can be solved by reading the last applicable obstruction receipt. V42 requires combining two independently updated remote actuators A and B. The remote junction becomes blocked only if both A=1 and B=1, implemented as a physical adjacency-graph change in the synthetic Host. Environmental actuator changes are host events, not policy actions.

In each matched positive pair, A's final receipt is identical and later than B's final receipt. B's value is the only causal difference, at least 112 accepted movement actions before the decision; A's latest receipt is at least 80 movements old. Legal actions, local geometry, recent 64 action and position histories, and sensor diagnostics remain equal. The first optimal movement is MOVE_E when B=0 and MOVE_S when B=1 (given A=1). Both first movements are legal in both worlds; choosing the other route incurs at least two extra accepted actions. The theoretical first-action ceiling with B-only receipts removed is 50%, not a neural-model measurement.

Negative controls set final A=0. B changes the physical graph earlier while A=1, but with A=0 at the decision both B values imply the same best first action. This prevents simply treating any B difference as an instruction to change policy.

## Generated corpus

| Split | Strict positive pairs | A-off controls | Episodes | Policy decisions | Topologies |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 128 | 32 | 320 | 8,912 | 8 |
| DEV | 64 | 16 | 160 | 8,736 | 4 |
| HELDOUT | 64 | 16 | 160 | 12,096 | 4 |
| Total | 256 | 64 | 640 | 29,744 | 16 |

640/640 oracle-driven host continuations ended successfully without residual incorrect paint. Every accepted step was checked against the exact reverse-BFS suffix oracle. Strict positive contrast and minimum regret were checked for both members of each pair. Trainer-visible policy rows contain only `input`, `action`, `weight`: present only `input` to the model; `action` is the teacher label, and `weight=0` excludes duplicate TRAIN inputs. DEV and HELDOUT challenge rows are evaluator-only.

## Local qualification and reproducibility

Two independent full CPU generations produced byte-identical five JSONL shards and manifests. Independent audit checked all 29,744 policy rows and all 160 DEV/HELDOUT evaluator pairs. Positive and negative cases were independently reinstantiated across all three splits and four schedules. No conflicting labels, exact-input duplicates or cross-split geometry overlaps were observed.

- Generator source SHA-256: `e08d8c186b54b4680a59c33169ec150f0873b4f309303a57ef909ad01449d993`.
- Independent test SHA-256: `12e622126cbb1509df5f3183ff48d021117c7476226911c3891825a4d3309933`.
- Canonical generated manifest SHA-256: `fce49195cdfae48ec02a71928a85fecfe398f4e3c93e8b2aefa0fa0ff5af2834`.
- Two measured generator runs: 13.89 / 13.78 seconds; peak RSS 490,228 / 439,532 KiB. These are generator measurements, NOT SLM inference latency or model RSS.

With Node.js and the pinned dependency files already in this same directory, reproduce outside the Git working tree:

```bash
node dual_interlock_v42.js /tmp/driv3r-v42-a
node dual_interlock_v42.js /tmp/driv3r-v42-b
node test_dual_interlock_v42.js /tmp/driv3r-v42-a /tmp/driv3r-v42-b
```

Compare the generated manifest with `V42_DEFAULT_MANIFEST.json`. Shard SHA-256, byte counts and row counts are pinned in that manifest. Generated JSONL shards and checkpoints are intentionally not Git-tracked; immutable Actions artifact publication is NOT complete.

## Limitations and consumer handoff

The challenge is a fixed synthetic symbolic junction, 16 topologies and Boolean AND logic. Deterministic scripted prehistory plus a symbolic state parser can solve it. Neither model adaptation nor general navigation capability has been established. There are no trained LFM/Sprocket weights from this lane, no actual SLM closed-loop HELDOUT results, and no native CyberSand validation or model latency/RSS evidence.

The model-training consumer should train a pinned actual SLM on TRAIN only, select on DEV, then execute HELDOUT autonomous host-play under: (1) full durable memory, (2) B-only event ablation retaining A and sensors, (3) last-64 history. Stratify strict A-on positives versus A-off negative controls. Report terminal success, residual contamination, action regret, illegal preference rate, first-action contrast, p50/p95 inference latency and peak model RSS, with model/tokenizer/adapter hashes. Validate native CyberSand separately against the canonical host-legal observation/action ABI. V38 remains the broader sequential tool-law training source.

No paid API/GPU spend, no out-of-lane writes, no changes to DRIV3R main, CyberSand or Neurounner. PR #1 remains Draft/HOLD; do not merge or mark Ready.
