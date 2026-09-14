# Help-wanted issue drafts

Prepared September 15, 2026. **Not posted:** the GitHub integration returned an issue-writing permission error, and the browser requires a one-time 2FA check. Repository source push succeeded separately. Complete verification on GitHub itself before posting; do not share authentication codes in issues or chat.

## 1. Help wanted: reproduce the archived release on an independent computer

Independent checks are more valuable than an untested claim of reproducibility. This is a small entry point for a first contribution.

Follow REPRODUCIBILITY.md on a clean environment. Record the commit, operating system, Python version, and dependency versions. Run `python scripts/validate_release.py` and report exact commands and output, removing secrets and personal details. Checkpoint evaluation and fresh training must be identified separately.

Done when an independent contributor posts the environment, commands, and observed results, whether success or failure. Release validation needs no GPU and checks archived counts/checksums; it is not a full scientific replication. Use the reproduction issue template. Contributors will be credited with their permission.

## 2. Help wanted: audit quantity identity and equation applicability

Can the graph distinguish the same type of quantity in different physical contexts?

Inspect `cm_fullgraph_builder.py` equations EQ_rho_mV, EQ_m_rhoV, and EQ_buoy, together with the fluid scenario in `scenario_generator_v1.py`. Shared Q_rho and Q_V_vol participate in mass and buoyancy expressions. Determine when the mass represents displaced fluid rather than the object itself, and document required assumptions.

Deliver a short valid-scope explanation, a concrete counterexample to any overbroad interpretation, and a proposed representation distinguishing object identity, time, units, and reference frame. Suggest equation applicability conditions. No retraining is needed for this first review. Preserve published artifacts; corrected experiments must be separately versioned.

The live symbolic companion is https://physics-route-lab-minnatullah.ayan14.chatgpt.site. It is not a GNN reasoning trace. Criticism is welcome; no new-physics claim is being made.

## 3. Help wanted: review an independent pendulum measurement protocol

Start by reviewing REAL_DATA_ROADMAP.md, not by buying equipment. Propose a safe low-cost setup, with measured lengths, repeated timing trials, units, instrument details, and uncertainty. A small pendulum with video timing is an option. Never risk an unsecured phone or stand near an unsecured swinging mass.

Include raw timing and length records, calibration, uncertainty, trial identifiers, and a repeatable procedure. Specify small-angle and effective-length assumptions; an extended phone pendulum is not an ideal point mass. Reserve whole lengths or an independently operated setup for evaluation. Compare against a direct known-formula baseline before introducing a learned model.

State the permission/license for sharing data. Do not upload personal identifiers, precise home locations, or incidental recordings of people.

Done when a protocol and data schema are reviewed. This issue does not claim measurements already exist, and the proposed study tests recovery of known physics, not a new law.
