# Validation — September 15, 2026

- Ten calculation tests passed, including all 64 observation masks across six possible targets, target exclusion, indirect paths, invalid values, inconsistent inputs, and insufficient information.
- Main UI supports four targets: mass, net force magnitude, momentum magnitude, and kinetic energy.
- JavaScript syntax checks passed. The entrypoint and assets were served successfully over local HTTP.
- Chrome UI inspection confirmed the default four mass routes, three remaining routes after hiding force and acceleration, and an explicit 2–8 kg disagreement warning after changing momentum from 8 to 16 kg m/s.
- Desktop layout was visually inspected. Mobile breakpoints are implemented; a dedicated mobile-browser visual inspection was not performed.
- Optional WebMCP tools are feature-detected. A supported tool-execution context was unavailable for end-to-end WebMCP validation; no such validation is claimed.
- The seven existing research-release checks passed unchanged. No training run, real-data experiment, or new discovery audit was performed as part of this demo release.

The tolerance used to flag route disagreement is a UI convention, not calibrated experimental uncertainty. The demo has no visitor analytics; publication alone does not establish any audience or adoption.
