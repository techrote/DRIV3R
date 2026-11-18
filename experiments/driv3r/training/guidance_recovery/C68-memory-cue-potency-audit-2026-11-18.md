# C68 — Memory-corruption potency and cue-strength transfer audit (2026-11-18)

**Disposition: HOLD.** Evidence review of previously executed C65/C66/C67 synthetic-host suffixes, not new training, native gameplay or an independent confirmation trial. Research ownership: DRIV3R PR #1 only.

## Question and method

Do the previously reported 0% versus 50% historical-memory-loss arms actually produce different *host-executed actions and painting* for the frozen recovery policy, and is stronger backtrack guidance consistently advantageous? Recompute from the original committed compressed ledgers using \`c68_memory_cue_audit.py\`. Those ledgers match the owner archives' Git blob SHA identities; no source ledgers were edited.

Graph identities are disjoint: C65 12, C66 24, C67 36 (72 total). Match within graph, policy, cue strength and memory arm. Compare action SHA-256 *and* ink SHA-256, not merely completion counts. Calculate graph-clustered exploratory intervals on paired strong-minus-weak completed goals; do not count two memory arms as independent graph replicates.

## Key finding: weak potency of the memory-erasure comparison

For the frozen C60 neural controller, **140 of 144** graph × cue-strength comparisons (0% versus 50% memory erasure) produced **identical action and ink hashes**:

| Cohort | Weak cue (.35) | Strong cue (1.0) | Frozen controller total |
| --- | ---: | ---: | ---: |
| C65 | 12/12 identical | 12/12 identical | 24/24 |
| C66 | 23/24 identical | 22/24 identical | 45/48 |
| C67 | 36/36 identical | 35/36 identical | 71/72 |
| **Combined** | **71/72** | **69/72** | **140/144** |

C65 and C66's C18-A0 and C34-RESET comparison policies also had exactly identical action/ink traces under both dropout arms for every graph and cue strength. C67's trained ALIGNED gate selected outcomes with matching action/ink hashes in 33/36 graph comparisons and its SHUFFLED control in 31/36. No frozen C60 strong- or weak-cue completed-goal outcome changed between 0% and 50% erasure in these cohorts.

**Interpretation:** The corruption intervention was behaviorally almost inert in these remote-goal episodes. This does *not* prove that the erasure code malfunctioned: if no relevant stored route was consulted, correct erasure may not alter the policy. It does mean that nearly identical outcome rates across the two memory-loss arms **cannot be treated as strong empirical evidence of resilience to consequential memory damage**. Retain the original counts, but qualify that claim.

## Cue-strength effects change sign across cohorts

Intact-memory frozen C60 neural policy, paired within each independently generated graph:

| Cohort | Weak goals | Strong goals | Strong-only wins/losses | Difference | Descriptive graph bootstrap interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| C65 (12) | 5 | 7 | 2 / 0 | +2 | [0, +5] |
| C66 (24) | 14 | 13 | 0 / 1 | −1 | [−3, 0] |
| C67 (36) | 17 | 21 | 4 / 0 | +4 | [+1, +8] |

Stronger guidance increased novel-site counts in all three cohorts (C65 91→127; C66 268→316; C67 244→451), yet worsened completion on one C66 graph. Hence C67's stronger-cue superiority does not establish universal nonregression. The C66 cohort was used to train the C67 selector, so it is not independent held-out confirmation of selector quality; the C67 cohort was fresh for that selector.

## Next experimental gate

Do not train another selector on a largely vacuous erasure contrast. **Before model training**, predeclare memory-rich graph prefixes with independently verified route entries, record pre/post memory-entry hashes and removed-route counts, and preserve a separately reserved family-balanced held-out set. Measure both the manipulation's observed read/use and *post-intervention* behavioral divergence without selecting success cases by outcome. Evaluate no-guidance, weak and strong cues against C18-A0, RESET, non-neural FRONTIER and frozen C60; report paired regressions, exploration and full-path personality metrics. Use equal host-action exposures, matched optimizer work and wrong-credit controls for any new trained head.

## Reproduction, custody and limitations

From \`experiments/driv3r/training/guidance_recovery/\`, run:

\`\`\`bash
python -W error -m unittest -v test_c68_memory_cue_audit.py
python c68_memory_cue_audit.py . /tmp/C68-summary.json
\`\`\`

Three immutable input Git blob SHAs:

- C65 ledger \`8152f4b32237a4cb68d377f800401145b884ac73\`
- C66 ledger \`4c9e5ea53e429f5389452fc348bade3bdc7e0719\`
- C67 ledger \`504a18ac6366b535162c29c0d923d6fa5a11cde4\`

Local audit: 6/6 tests passing; all 72 graph identities unique across cohorts. The analysis is post hoc, graph-cluster intervals are descriptive, and C67 selector outcomes use precomputed executed counterfactual suffixes rather than independently executing each selection. The audited host is synthetic, not the authoritative CyberSand runtime. No new optimizer, model checkpoint, native acceptance or immutable Actions checkpoint custody is claimed.
