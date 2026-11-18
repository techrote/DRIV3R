# DRIV3R Corpus Factory v37 — live material-tool-law counterfactuals

18 November 2026 · `techrote/DRIV3R` PR #1 Draft/HOLD · **Synthetic off-host decision forks; NOT native CyberSand, and NOT end-to-end autonomous trajectories**.

## Why

The v35 corpus has valid long-history painting recovery, but the same fixed correspondence (`PAINT_A` produces A, `PAINT_B` produces B) throughout every episode. It is possible to obtain low painting-token error by learning that correspondence rather than responding to a changed environment law. V36 separately probes a causally meaningful *goal quota revision*; v37 probes the independent ability to **condition painting commands on a currently installed mapping** without requiring any new policy learner.

## Intervention and exact scope

For every selected v35, host-accepted decision state, the synthetic environment installs one of two **public, future-only** material-tool laws just before the decision:

| Live law | `PAINT_A` physical result | `PAINT_B` physical result |
|---|---|---|
| normal | A | B |
| inverted | B | A |

The public `input.law` discloses the active mapping. Historical PAINT receipts are **not rewritten**: all were made before this newly installed rule and retain their original meaning. The two matched inputs are otherwise byte-identical. Both commands remain in the same host-legal action menu at paintable, empty target cells.

This is a *counterfactual, one-step state fork*. The teacher action is derived by the exact action-token bijection that preserves the resulting material effect. Navigation, ERASE, and STOP are unchanged-law controls. This is an honest **synthetic tool semantics** test; it does not assert that native CyberSand has such a law, and does not independently rerun v35's entire synthetic host from the intervention forward. Future action-token remapping preserves physical semantics, but the study must not be presented as a newly collected autonomous closed-loop episode.

## Selection and empirical evidence

Deterministically select every `PAINT_A`, `PAINT_B`, `ERASE`, and `STOP` decision plus navigation controls selected by `uint32(SHA256(input)[0:8]) % 12 == 0`. Each state contributes both laws. TRAIN carries v35's loss weights, with repeated complete transformed inputs downweighted to zero; DEV and HELDOUT retain all rows as evaluation data. The policy JSONL records contain exactly `input`, `action`, and `weight`; the paired challenge records, with labels, are kept in separate evaluation-only shards. Do not expose whole rows or counterfactual records as prompts.

| Split | Source state pairs | Paired paint-action changes | Unchanged controls | Policy rows | Nonzero loss weights |
|---|---:|---:|---:|---:|---:|
| TRAIN | 5,920 | 1,792 | 4,128 | 11,840 | 10,020 |
| DEV | 3,641 | 896 | 2,745 | 7,282 | 5,872 |
| HELDOUT | 4,090 | 896 | 3,194 | 8,180 | 6,514 |
| **Total** | **13,651** | **3,584** | **10,067** | **27,302** | **22,406** |

With the public law field ablated, a model must produce the same prediction for the two otherwise identical inputs. Consequently its exact maximum first-action accuracy on **paint-only paired challenges is 50%**, regardless of model architecture. On unchanged-action controls, the correct labels match; these are explicit negative controls. No neural accuracy is claimed.

A frozen-normal action in an inverted paint-positive context yields the opposite physical material; conditional on completing the same required cleanup, a wrong initial paint must be erased and replaced, adding at least two synthetic actions. This is a symbolic lower bound, not a measured rollout regret.

## Qualification and provenance

- Source: exact v35 trainer-safe manifest SHA-256 `db19fb6e565d443f3df494c1b5b64fd332c34d54a7048e2cef58a02d55e6b3f9`. Generator refuses parent shard hash/size mismatches.
- 16 TRAIN, 8 DEV and 8 HELDOUT disjoint source graph topologies; zero cross-split graph or full-input overlaps, zero contradictory labels.
- 27,302 generated policy records, complete model/teacher-field audit, legal-action assertions, independently rechecked target recipes and output-law mapping for all 3,584 positive pairs, and zero false-positive action changes among 10,067 negative pairs.
- Independent full-data SHA-256 audit; two complete generations produced byte-identical six output shards and manifests.
- Local generator runs: 3.18 and 3.09 s, peak RSS 90,080 and 96,232 KiB. Independent audit: 3.70 s, peak RSS 154,812 KiB. **Not SLM latency/RSS.**
- The generator and local tests use Node.js only. Run `node material_law_forks_v37.js /path/to/verified/v35 /tmp/v37a`, repeat with a fresh destination and then `node test_material_law_forks_v37.js /tmp/v37a /tmp/v37b`.
- `V37_DEFAULT_MANIFEST.json` lists every source/shard checksum and all partition metrics. Generated corpora belong in immutable opt-in Actions artifacts or a local release archive, not Git. No automatic training on ordinary pushes.

## Negative results and next consumer experiment

**No model weights updated, no new neural inference, no native ABI evaluation, no closed-loop counterfactual host acceptance, and no pretrained root-model binaries available in this data lane.** V37 inherits v35's simple radial topology and is a targeted material-token-law discriminant, not an adequate substitute for v36's goal-change/remote-memory task.

A consumer should first train a pinned actual LFM/Sprocket adapter on TRAIN only and evaluate HELDOUT material-positive pairs **separately** from unchanged-law controls. Compare mapping-visible against mapping-ablated input and report accuracy, wrong-material rate after live host execution, successful correction, model latency/RSS and topology-generalization. Combine this with v36's delayed goal change and off-policy recovery for the next true sequential-law-change challenge. No unsupervised new policy trainer is authorized in this data lane.
