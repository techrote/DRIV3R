# DRIV3R research work rules

Repository ownership: **techrote/DRIV3R** is the only writable repo for DRIV3R experimental models, training, research curricula, model zoo, population agents, model surgery and result ledgers.

- CyberSand and Neurounner are **read-only references** to DRIV3R research workers. Never open, edit, close, merge or comment on their PRs; never push or change files there.
- CyberSand retains gameplay/runtime ownership: its native/WB worm controls, host authoritative action/observation contracts, integration tests and real game evaluations. Do not copy or modify game/Workbench/runtime source in this repo. Source ABI snapshots here are *historical fixtures*, not contracts of record.
- Neurounner retains unrelated neural experiments, notably DREAMGRAFT/MEMPLANT.
- The initial import is carried through **one Draft/HOLD PR**: branch \`research/import-consolidated-worm-labs\`. Do **not** create new per-experiment PRs or issues. Research workers should append to existing bounded programme paths/ledgers in that branch, unless explicitly instructed otherwise.
- Check the exact remote branch SHA before editing. A concurrent worker may have committed. Change **only your owned lab subdirectory**, fetch and replay onto the latest head with fast-forward-safe pushes, never force-push or rewrite another worker's files.
- Preserve pinned model revision IDs, actual optimizer/inference receipts, evaluation seeds, artifacts, source-hash provenance, negative results and methodological caveats. Never report script/fixture outputs as trained-model successes.
- No merge, no Draft-to-Ready, no automatic default behaviour change, no paid/API/GPU spend without authorisation. Time/compute-bounded training is explicit and manually initiated, never scheduled on pull_request or push.
- Keep the repo clean: one consolidated research PR, minimal CI, avoid one-off workflows for each minor ablation, no copied model weights or bulky binary artifacts committed to Git.
