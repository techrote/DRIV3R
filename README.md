# DRIV3R

**Experimental autonomous-agent and small-model research.**

DRIV3R is the independent research home for game-playing and environment-interacting model agents: worm controllers, compact game policies, model surgery, recurrent/SSM architectures, quantisation, distillation, synthetic world curricula, inference evaluation, multi-agent populations and research-fleet orchestration.

## Repository boundaries

- **DRIV3R (here):** models, training/evaluation, synthetic worlds, research corpora, experiments and reproducible evidence.
- **[CyberSand](https://github.com/techrote/cybersand):** native simulation, Workbench, runtime action/observation integration, real gameplay testing, performance and owner acceptance.
- **[Neurounner](https://github.com/techrote/neurounner):** broader neural research such as DREAMGRAFT/MEMPLANT. It is not the long-term owner of DRIV3R research.

## Research provenance

The initial source is being imported, with Git blob verification and original experiment/evidence references, from [Neurounner staging PR #252](https://github.com/techrote/neurounner/pull/252). That staging PR had consolidated the displaced CyberSand model-research branches without merging them. The original PR histories, scientific caveats, checkpoints and hosted Actions artifacts remain referenced in the research manifests; **external artifacts are not assumed transferred**.

The migration is source preservation, **not** a claim of independent scientific reproduction, native integration or model quality. Experimental outputs must distinguish actual trained/inferred results from fixtures and scripted controllers.

## Contribution and merge policy

Use a small number of coherent long-lived research branches and PRs, not a PR for each run. Keep CI and inference budgets explicit; do not launch costly training automatically on routine pushes. Never merge, promote Draft PRs to Ready, modify production game code or assert physical hardware acceptance without the owner's explicit approval.
