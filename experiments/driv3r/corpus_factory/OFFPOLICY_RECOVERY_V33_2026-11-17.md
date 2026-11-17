# DRIV3R Corpus Factory v33 — accepted off-policy mistakes and full recovery

**2026-11-17 · TRAIN/DEV/HELDOUT · synthetic host, NOT native CyberSand · Draft/HOLD PR #1 · data lane only**

## Question

Can a corpus preserve v32's four-way old-memory counterfactual *and* supervise recovery when the host actually accepts a detrimental action (as opposed to recording an unexecuted negative preference)? The downstream trainer needs legal feedback and a true optimal recovery suffix, not a hand-coded RETRACE rule.

## Implementation

- Pin `causal_shortcut_v32.js` plus v26/v25/v30 dependencies and their exact SHA-256s in the manifest. Use v32's topology-disjoint four-way families and validated five-target status-by-position reverse-BFS suffix oracle.
- For each of four old-error locations, generate **two** entirely host-accepted continuations: (A) optimal clean and (B) wrong-first recovery. The latter accepts a **legal, non-optimal navigation action** selected deterministically among finite alternatives with at least two steps of regret; the wrong action is a genuine `host.apply`, with an action acknowledgement in the new initial history. Subsequent labels come from freshly queried shortest-successful-suffix values at every resulting state.
- Attach exhaustive legal-action cost/regret certificates as **teacher-only** decision fields. Only `state + reconstructed history` or `bounded_model_input` are permissible observations. Neither certificates nor `hidden_interventions`, `first_action_required`, `episode.mode`, `split`, or `family_id` may be passed to a policy.
- Keep the original v32 strict closed-gate versus opened-gate causal intervention, old painting-error memory alias, material correctness and protected cells. Two erasures are required in every trajectory.
- Persist six JSONL shards with compact initial-history-plus-delta replay, exact observation/bounded-view hashes, split identity, and shard SHA-256 manifest. Shards are generated outside Git and are **not** falsely claimed to be immutable Actions artifacts.

## Full deterministic generation

| Partition | Four-way families | Episodes (clean / bad-first) | Expert-labelled decisions | Accepted bad-first actions | Corrective erasures | Realised regret |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 64 | 512 (256 / 256) | 14,912 | 256 | 1,024 | 512 |
| DEV | 32 | 256 (128 / 128) | 12,544 | 128 | 512 | 256 |
| HELDOUT | 32 | 256 (128 / 128) | 17,024 | 128 | 512 | 256 |
| **Total** | **128** | **1,024 (512 / 512)** | **44,480** | **512** | **2,048** | **1,024** |

All 512 accepted bad first navigation decisions have **exactly 2 shortest-suffix actions of realised regret** against their matched optimal clean continuation. The 512 clean episodes retain the v32 clean first-action four-way `BACKTRACK`, `MOVE_W`, `MOVE_E`, `MOVE_S` balance: among each matched family, recent-64 and no-durable-memory inputs are identical, yielding a 25% theoretical uniform first-action ceiling. The 128 clean east-arm instances retain strict causal open-gate navigation regret; other 384 clean trajectories are explicit gate-negative first-action controls.

## Independent qualification

The v33 test reconstructs each compact observation solely from episode-initial history and decision deltas, recomputes both its SHA-256 and compressed-view SHA-256, replays **all** host actions and checks accepted receipts, reruns exact five-ternary-target reverse BFS for each episode, verifies **all** legal action cost/regret certificates, enforces optimal post-intervention suffix length and terminal cleanliness, verifies two erasures per trajectory, and checks four-way aliases, cross-split input consistency, geometric split separation, shard hashes and dependency pins.

- **1,024/1,024** host replay terminal successes; **44,480/44,480** expert decisions replayed and hash-checked; 512/512 accepted wrong-first interventions; **0** conflicting complete-input labels; **0** cross-split input or graph-geometry hash overlaps.
- **0** repeated full-input TRAIN records in this generation. No duplicate downweight was required; the generator retains defensive downweight logic.
- Two independent full generations: six JSONL shards plus manifests **byte-identical** (verified `cmp`).
- Generator A: **18.91 s, 796,356 KiB peak RSS**; generator B: **19.33 s, 1,002,600 KiB peak RSS** (CPU/environment variation; **not neural inference measurements**). Local test also separately passed both generated corpora.
- Default manifest SHA-256: `f006848f5b7bfc78148a469e37e5de70a152650f3f80928efd802d48da436351`.

## Negative evidence and limitations

This is an **oracle-labelled synthetic data benchmark**. It does not demonstrate SLM improvements, genuine autonomous exploration, model-led gate triggering or native CyberSand mechanics. The past wrong paints and gate-opening paint remain host-accepted but scripted; only the **new detrimental navigation intervention and its full recovery** are exercised after the v33 split point. The topology grammar is radial and small, the material law is static, and an explicit interpreter can infer older error receipts. Regret certificates must not leak into model inputs. No model checkpoint, optimizer update, GPU/API spend, native test, or real-model latency/RSS result is implied. No immutable Actions artifact was produced.

## Next useful consumer evaluation

Pin an actual pretrained LFM350/Sprocket root and adapter by hash, run bounded CPU finetuning on TRAIN and DEV only, then closed-loop HELDOUT with **no teacher certificates at inference**. Compare model-controlled clean and accepted-wrong-first recovery with full durable history, recent64-only, last-PAINT-only and memoryless controls. Report terminal success, residual wrong material, realised regret, premature STOP/RETRACE, legal preference rate, p50/p95 step latency and process peak RSS. Keep checkpoint binaries in immutable Actions artifacts and never promote from synthetic results directly to CyberSand.
