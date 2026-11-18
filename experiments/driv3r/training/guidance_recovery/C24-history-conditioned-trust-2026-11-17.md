# C24 — Audited source-history cue trust (17 November 2026)

**Disposition: negative for promotion. Keep C12 as frozen trained reference and C18-A0 as the stronger untrained attestation control. Draft/HOLD; no merge.**

## Question and experimentally observed answer

Can a bounded adapter decide whether to heed host `BACKTRACK` or `CONTINUE` guidance using *past host-executed feedback*, instead of treating the latest directive as a command? C24 trains a 8→12→1 head on a frozen C18-A0/C12 predecessor. Historical evidence is obtained from six paired, completed synthetic calibration episodes per advice source; each historical episode compares the actually host-applied `BACKTRACK` branch and best legal forward branch with matched random seeds. The query's hidden outcome, future geometry and latent source-quality regime are not supplied to the model. Only four normalized audit summaries append to the original 32-feature research observation; the native CyberSand interface is untouched. Uncued behaviour and all non-BACKTRACK scores are exactly frozen.

A source has a latent propensity to issue nominally correct advice (80%) or misleading advice (20%). The latent mode is **only a dataset-generation variable**, not a model feature. Training mixes four outcomes (goal behind, goal ahead, shortcut and bridge), two personalities, three spatial-hint strengths and real on-policy counterfactual host-executed suffix returns. Evaluation withholds hook/zigzag geometry and source seeds; it includes fully crossed adversarial guidance and naturally source-issued guidance with no/weak/strong arms. A separately optimized no-history head and a wrong-source-history ablation test whether memory actually carries useful predictive information.

**Result:** although the history-conditioned head improves the optimizer objective and slightly restores preferred-INK probability on matched active menus, it loses real goal completions against C18-A0. The wrong-source control performs similarly, suggesting weak causal value from this particular audit history representation. Historical audit signals are noisy: in the first held-out cohort, the nominally reliable sources still had mean audit reward −0.186. No statistically strong or promotion-worthy history-conditioned trust has been demonstrated. Do not generalize this synthetic result to native CyberSand physics or production behaviour.

## Executed evidence and baselines

Each of two independently seeded held-out cohorts contained 5,040 executed episodes across five policies. The common 288 training menus produced 1,152 host-executed counterfactual suffixes; each new head received 90 CPU AdamW updates (180 total). Training repetitions produced byte-identical model checkpoints and losses. All 162 locally available offline tests passed (156 inherited plus six C24 tests). No illegal raw top-choice actions, synthetic host-rejected commands or protected-material changes were seen.

| Model | Fully crossed adversarial guided goals | Source-issued guided goals |
|---|---:|---:|
| C12 | 910/1152 | 453/576 |
| C18-A0 | 945/1152 | 475/576 |
| C24-TRUST | 939/1152 | 471/576 |
| C24-NOMEM | 936/1152 | 470/576 |
| C24-SHUFFLE | 934/1152 | 469/576 |

The strict non-regression gate checks each topology/outcome/direction/strength group against C18-A0, not just aggregate completion. C24-TRUST failed 9 groups in cohort A and 6 in fresh-seed cohort B; C24-NOMEM failed 11 and 7; wrong-source C24-SHUFFLE failed 13 and 7. These groups are correlated and small; counts are descriptive, not p-values.

Cue-off controls reproduced C12's action and ink hashes exactly, and every candidate changed only BACKTRACK scores when a cue was active. The measured same-state personality-favoured INK probability rose from 0.2780 to 0.2801 in crossed active-guidance menus and from 0.2906 to 0.2927 in source-issued active-guidance menus (C18-A0 → C24-TRUST). The improvement is small and did not compensate for lost recoveries. Genuine source-issue advantage was not established.

## Source, checkpoint and scientific caveats

- Repository target: `techrote/DRIV3R`, existing Draft/HOLD PR #1 only; parent branch `research/import-consolidated-worm-labs`.
- Inputs: C23 historical research sources in the owned directory; frozen C12 safetensors from the C23 archive, exact SHA-256s preserved in the external `metrics.json`; C18-A0 untrained host-attestation gate retained.
- Program: `c24_history_trust.py`; tests: `test_c24_history_trust.py`; verified gallery generator: `c24_gallery.py`. Original code remains in the owned directory; checkpoints, original full training/evaluation receipts and SVG trajectory gallery are external interim artifacts, not committed binary blobs.
- Replay gallery: 20 panels selected from the observed divergent trajectories, **replayed against the exact source and compared against recorded action/ink hashes**. All matched.
- This experiment uses a synthetic host with a replay/audit facility; this facility is not a proposal to modify CyberSand's native ABI. The old `d_common` fixture is not authoritative for current game legality. Qualifying native, physical gameplay remains a separate owner gate.
- Source-history calibration is not causal evidence of advice quality on an unseen target. Source modes are generator labels and should not be equated with true utility on every map. Some so-called nominally correct directives are unhelpful due to shortcuts and partial observability.
- Complete GitHub Actions scientific reproduction and immutable Actions custody of checkpoints have **not** been achieved. Checkpoint SHA receipts below are interim local custody; do not claim cloud artifact preservation.

## Exact hashes

- `c24_history_trust.py` SHA-256 `de8fb151d7fdd80b192b65ea48b121905986fdbf6b5a9068f933ac1b8146bb8a`
- `test_c24_history_trust.py` SHA-256 `bd6afc56aed952acbb240a2292006bda23d8b86f996b7b500a925648902ae4b7`
- `c24_gallery.py` SHA-256 `eb7f062e1bcadf5ad88c80e7012ac135ed9c56e2f5dff4e68c4e332385100cf9`
- Training dataset SHA-256 `7314bfd00df73a2f58f26277d58e21d46d39a3b235b2f4558f6748992d1df3ab`
- C24-TRUST checkpoint SHA-256 `5ae727d92508a7975e7b83eded444b5964602dfb42329cd5948b3f05e663c412`; AdamW objective -2.36e-05 → -0.0027324
- C24-NOMEM checkpoint SHA-256 `7d0f48f4cbceaff0b4a18834b7a775a6b1d7ec52b0a664741bf6b42d7a851c22`; AdamW objective -1.74e-05 → -0.0025336
- Held-out cohort A seed `241999`, metrics SHA-256 `bd17e024aa05bc3f23a5e8873b58cec90f2ad9e2bd031492392f51cf58b7c2a3`
- Held-out cohort B seed `242999`, metrics SHA-256 `7275ae646018ac0543633476dae868b2737e1d7bad798ee51880d60b7c63cc52`

**Next research question:** determine whether a replay-free, online verifiable feedback signal (achieved correction versus repeated stuckness/revisit) predicts future cue utility on withheld native-like games. Reject methods that depend on a hidden source-quality variable or that only improve the optimizer objective without better trajectories.
