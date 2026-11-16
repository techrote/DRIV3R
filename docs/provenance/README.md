# Research import custody

This directory contains a machine-verifiable snapshot of a specific commit from Neurounner's DRIV3R staging PR #252, whose files were originally recovered from CyberSand research branches.

\`SOURCE_SNAPSHOT.json\` is created by the one-shot source importer and records:
- exact original Neurounner source commit and PR;
- every imported source path and destination path;
- byte count and Git blob SHA-1, checked before and after copying;
- per-file accounting against the four CyberSand migration manifests;
- any historical research files that advanced after the initial CyberSand migration.

The files in \`original-migration-tools/\` are archived copies of the original CyberSand-to-Neurounner migration tooling for archaeology. They should not be rerun on the permanent repository.

This provenance verifies **source integrity only**. Checkpoints, actual model weights, artifact ZIPs and original test executions are not imported. Original public PRs remain the authoritative evidence source until separately reproduced.
