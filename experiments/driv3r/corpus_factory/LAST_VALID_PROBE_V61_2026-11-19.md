# Corpus Factory v61 — last valid physical probe, independently crossed old/new laws

Date: 2026-11-19. Repository: techrote/DRIV3R. Programme PR: #3 (Draft/HOLD). This is **DATA and host-executed synthetic evidence**, not trained policy improvement and **NOT native CyberSand**.

## Research question

Can a policy ignore obsolete physical evidence, correctly retain the **latest** material-law probe after a later non-diagnostic warning, and select the protected material after a long period without event-history replay? V60 proves memory of a single historical B probe; v61 independently varies both the **earlier** and **later** observed B law. The final instantaneous observation (including legal actions, recent actions/positions and one-shot event delta) is byte-identical across each four-arm quartet, yet two arms require `PAINT_A` and two `PAINT_B`. Historical B probe outcomes and their temporal relationship to the two warnings are available only once in the sequential inputs.

## Intervention and labels

For each previously split-owned v46 world geometry (indices 700+), the v53 synthetic host runs A and enters B. It executes the first B safe probe (`PAINT_A`), physically observes material A/B, then erases the paint. On the return path a **second externally injected B SURFACE_ALERT** is delivered; the B law may remain stable or change, independent of the old law. The teacher must travel back, physically perform another B safe probe, erase it and return to the protected objective. A fixed-length legal movement/retrace intervention ages the latest receipt before the final material decision. All forced detour moves are `weight=0`, while both safe calibrations, safe erases and final goal paints remain labelled host actions. No new model action token or host ABI field was added.

The crossed arms are **old normal/new normal**, **old normal/new swapped**, **old swapped/new normal**, **old swapped/new swapped**. Warning presence is held constant. All four final instantaneous observation hashes match. The second physical outcome determines the final paint. Old-only and current-input-only critical-action heuristics have 50% ceilings by construction. A symbolic latest-receipt oracle reaches 100%; these are not neural accuracy measurements.

## Qualified full build

| Split | Worlds | Quartets | Successful host episodes | Policy decisions | Masked detour movements | Stale-old misleading |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 12 | 24 | 96 | 9952 | 6912 | 48 |
| DEV | 6 | 12 | 48 | 7680 | 6144 | 24 |
| HELDOUT | 6 | 12 | 48 | 13000 | 10752 | 24 |
| **Total** | **24** | **48** | **192** | **30632** | **23808** | **96** |

The independent executor imported **none of the v61 generation code** and reconstructed all 30632 pre-action inputs and legal/physical host action outcomes, including the old and new safe-probe receipts, event-age properties, incorrect-material negative forks, all protected-goal completions, the latest-event requirement, and TRAIN/DEV/HELDOUT geometry isolation. **48 / 48 quadruples** satisfy byte-identical final current inputs and opposing command causality. Source generation repeated twice produced byte-identical manifests and 6 JSONL files. A deliberately corrupted shard was rejected by the independent hash gate. No native or neural evaluation was performed.

## Reproduction

```bash
node experiments/driv3r/corpus_factory/last_valid_probe_v61.js /tmp/v61
node experiments/driv3r/corpus_factory/test_last_valid_probe_host_v61.js /tmp/v61
node experiments/driv3r/corpus_factory/last_valid_probe_v61.js /tmp/v61-second
# Compare SHA-256 across the two manifest+JSONL output directories.
```

Source SHA-256 (generator): `3c7b55ff5d462173505c0e82b1c01697adf66c56a64b671aa5eb6a05856005ab`. Independent auditor SHA-256: `8c831d389325657b36d05e54fa14a0b914ea8634ad4f3b907bcf5bc0e1c5e94d`. Generated full manifest SHA-256: `cdd3320c46c06e2b918fee06e1198f7ff24a78d7a711e2deb61f31792129df59`. Parent dependency content hashes are inside manifest, along with per-shard SHA-256 and lengths. Do not use the manifest's evaluator sidecars during training.

## Independent review / limitations

1. This is a synthetic graph host, with a **synthetically injected second alert**; there is no evidence CyberSand itself emits this alert or supports identical protected-material semantics. Cross-check only against CyberSand's **read-only** documented host-legal ABI before any real gameplay consumer work.
2. Artificially long detours are host-executed but **not expert route decisions**; all intervention movement targets are loss-masked. They do not establish learned long-horizon navigation.
3. A symbolic last-alert/last-probe memory controller can solve the task; this corpus is **a discriminative held-out causal challenge, not neural success**. Future trainer comparison must use a pinned actual LFM/Sprocket checkpoint/adapter and measure autonomous HELDOUT completion, memory resets and truncation sensitivity.
4. Geometry families come from the existing v46/v53 synthetic world generator; the 12 HELDOUT quartets are focused diagnostic evidence rather than broad real-world generalization. Cross-version dataset mixing must group by `geometry_sha256` and source seed.
5. No actual target SLM trained, no neural CPU p50/p95/RSS, no native CyberSand trial and no immutable Actions dataset artifacts published in this data lane. Raw corpus is reproducible and may be carried as a separate release/evidence ZIP without bulk data committed to Git.
6. Cross-lane trainer PR #2 remained untouched; current evidence indicates there are archived model/CPU replay experiments but no v61 consumer neural score. Do not promote an oracle score to a neural result.

**Consumer handoff:** add v61 TRAIN episodes as a small focused chronological-receipt curriculum alongside broad v54 TRAIN, v56 physically certified contrast and v59 recovery data. Never train on DEV, HELDOUT, or evaluator certificates. Evaluate autonomous quartet all-four-correct, stale-probe ablation vs latest-probe ablation, and frozen model latency/RSS. Keep PR #3 Draft/HOLD.
