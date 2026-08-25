# Contributing

Thank you for taking the time to examine this project. Reproduction reports are especially valuable because the central claim is auditability rather than novelty.

## Before opening an issue

1. Use Python 3.10 or newer and install `requirements.txt`.
2. Run `python scripts/validate_release.py` to verify the tracked artifacts.
3. Search existing issues before opening a new one.
4. Include your operating system, Python version, command, and complete error message.

Questions about scientific claims should identify the manuscript section, equation, artifact, or catalog row involved. Please distinguish a verified inconsistency from a proposed extension.

## Real-data collaborations

Collaborations are especially welcome from people who can contribute
independently collected mechanics, orbital, wave, fluid, or materials data.
Please describe the instrument or source, units, calibration, uncertainty,
license, and a leakage-resistant train/test split. A public dataset link alone
is not enough; the proposal should explain how its measured fields map to graph
nodes and how an external replication could be performed. See
REAL_DATA_ROADMAP.md for the staged protocol.

## Pull requests

- Keep changes focused and explain why they are needed.
- Do not replace published artifacts in place. New experimental artifacts require a new version and checksums.
- Run `python -m compileall -q .` and `python scripts/validate_release.py` before submitting.
- If the manuscript changes, confirm that `paper/physics_gnn_symbolic_discovery.tex` compiles successfully.
- Do not describe rediscovered consequences of the encoded corpus as new physical laws.

By contributing, you agree that source-code contributions are licensed under MIT and research-artifact or manuscript contributions are licensed under CC BY 4.0, consistent with the repository licenses.
