# C38 — Return-sign calibration and shuffled-label controls (17 November 2026)

**Research-only; reject for promotion.** Synthetic CyberSand-like legal-action environment, not a native CyberSand/Workbench acceptance result. C38 tests whether training a correction head to classify the *sign* of an executed backtrack-versus-forward return is more helpful than C37's regression head. Only experiments/driv3r/training/guidance_recovery/** is writable; PR #1 remains Draft/HOLD.

## Experimental intervention

The training input is C34's 9-dimensional current-host observation. For 80 independently generated tree graphs, the host executes two counterfactual eight-step suffixes, first BACKTRACK and first best currently legal forward action. Their actual executed return difference defines the label. Ties (38 of 320 examples) are excluded from binary training; all remaining 282 examples are retained. C38-SIGN receives aligned labels. C38-PERMUTED receives a fixed permutation of those labels, with identical model initialization, batch ordering, optimizer, step count and inferencing architecture. Both use 16 epochs, 144 CPU AdamW updates, and reproduce bit-for-bit on a second optimizer execution. The head produces a bounded ±1.5 BACKTRACK-logit adjustment and can only act through existing cue-gated legal-menu inference. No graph oracle or shortest-path field enters inference.

C37-LEARNED and C37-SHUFFLED remain unchanged checkpoint controls; C34-RESET and C18-A0 are controls. Exact source hashes, prerequisites and receipts are in C38-qualification-evidence.json. The full withheld-label rows, cohort episodes, and replicate checkpoints are retained in the SHA-256-identified local research archive, not in Git history.

## Withheld prediction (independent graphs, not training data)

91 withheld graph hashes yielded 362 executed-label states, of which 295 had nonzero return signs. Nonzero-sign accuracy was **75.59%** for C38-SIGN, **53.22%** for C38-PERMUTED, **70.85%** for C37-LEARNED and **53.22%** for C37-SHUFFLED. By family, C38-SIGN achieved 75.00% on tree and 76.42% on braid. The permuted head achieved 74.42% on tree but 23.58% on braid. This family reversal matters: the trained model is not just learning the majority class of both generators. In 4,000 descriptive graph-cluster bootstrap resamples, the aligned-minus-permuted accuracy difference was +22.37 percentage points (95% interval +10.0 to +34.4); aligned-minus-C37-LEARNED was +4.75 points (+0.34 to +9.09). These are *descriptive*, not multiplicity-adjusted confirmatory bounds.

The sign-trained model has *worse continuous return-value MAE* than C37-LEARNED (0.742 vs 0.549). It solves the narrower sign-prediction task, not return calibration in general.

## Held-out executed gameplay (three independent cohorts)

Matched conditions used two generators, two goal types, repeated goal changes at turn 50, both personality signs, full-strength BACKTRACK and CONTINUE cues plus no-cue controls, one guidance-arm strength and a 78-decision horizon. The smaller 48-guided-episode-per-policy total intentionally limits computational expense; do not interpret small count differences as conclusive.

| Candidate | A | B | C | Total guided goal events |
|---|---:|---:|---:|---:|
| C18-A0 | 6 | 4 | 9 | 19 |
| C34-RESET | 8 | 4 | 10 | 22 |
| C37-LEARNED | 8 | 4 | 9 | 21 |
| C37-SHUFFLED | 8 | 5 | 10 | 23 |
| C38-SIGN | 8 | 5 | 9 | 22 |
| C38-PERMUTED | 8 | 4 | 8 | 20 |

C38-SIGN ties C34-RESET, fails a condition-level nonregression gate in cohort C, and does not exceed the C37-SHUFFLED whole-trajectory result. The improved offline sign prediction therefore **does not demonstrate a gameplay improvement**. The strict gate forbids promotion. No new controller or action ABI claim is justified.

All policies retain their 8/8 matched no-guidance action and painting hashes in each cohort (24/24 per policy total); no synthetic host execution was rejected and no protected material changed. Whole-trajectory personality preservation remains unqualified: within-state projection is not a trajectory-level guarantee. C38's restricted horizon and low per-condition sample counts are a substantial limitation. Native CyberSand evaluation and immutable Actions checkpoint custody remain outstanding.

## Reproduction and next test

Run the C38 source with the unchanged C37/C36/C34 dependencies and SHA-identified prerequisite safetensors, or run targeted pytest checks. This is source-matched synthetic experimentation; it does not establish reproduction from a bare Git clone because artifact checkpoint custody remains pending. The most useful next experiment is an action-conditional policy-value learner on a genuinely distinct graph family, with graph-cluster-level power and trajectory-level personality auditing. Keep shuffled-label, random-head, and non-neural controls. Do not promote C38.
