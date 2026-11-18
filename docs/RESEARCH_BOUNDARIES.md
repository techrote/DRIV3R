# DRIV3R — ownership and migration architecture

## Permanent boundaries

**DRIV3R**: lightweight model architectures; pretrained SLM onboarding; cognitive/action research; policy training; dataset construction; quantisation, distillation and neural weight surgery; finite-horizon synthetic environments and ablation studies; model zoo, population policies, evaluation and training orchestration.

**CyberSand**: native cellular/material game engine; Workbench; game objects and actual worm movements; authoritative host input/output protocol, legal actions, renderer/camera and hardware acceptance. Model artifacts can be imported *from DRIV3R* into its game integration by pinned manifest; no research training or model zoo PRs belong there.

**Neurounner**: generally reusable neural research including MEMPLANT/DREAMGRAFT. This repo should not accumulate DRIV3R work after its staged source is migrated.

## What is migrated

The original CyberSand code was preserved in [Neurounner PR #252](https://github.com/techrote/neurounner/pull/252) as research-only source; its four exact-Git-blob custody manifests contain historical source PR, SHA and file inventories. The DRIV3R import is pinned to one Neurounner commit and copies only \`experiments/driv3r\`, \`experiments/agent_painter\`, \`docs/driv3r-migration\`, and archived import scripts. It preserves relative Python module paths and checks every destination file against its source Git blob. Details live in \`docs/provenance/SOURCE_SNAPSHOT.json\`.

The historical \`experiments/driv3r/d_common/v1\` protocol files are **read-only reference fixtures**. The CyberSand integration PR #594 remains the authoritative game-side contract.

Original hosted CI run artifacts, model checkpoints, archive ZIPs, evidence galleries, native game results and their original GitHub PR histories are **not copied**. The linked source PRs and original Actions records remain historical evidence, subject to retention; future reproducibility requires separately qualifying the new standalone environment.

## What is not claimed

A passing Git-blob audit and Python syntax checks establish that research source transferred exactly. They do **not** prove model training/inference reproducibility, scientific quality, checkpoint portability or successful CyberSand native integration.

## Repository and PR policy

The first branch is \`research/import-consolidated-worm-labs\`, Draft/HOLD against an intentionally small main. Maintain a small number of coherent programme PRs rather than generating one PR per trial. No merge without owner approval, and never automatically deploy models or alter the CyberSand Workbench.
