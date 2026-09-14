# Physics Route Lab

A small symbolic companion to [Physics GNN Symbolic Discovery](https://github.com/mi99at/physics-gnn-symbolic-discovery), by Md Minnatullah.

This is **not a deployed GNN**. The demo computes transparent routes over a small classical-mechanics equation graph. It does not expose the internal reasoning of the trained model, provide empirical validation, or claim new physics.

## Run and test

Serve `dist/` with any static HTTP server. No dependency installation is needed.

```sh
python -m http.server 8765 --directory dist
node --test tests/physics.test.mjs
```

The engine uses F=ma, p=mv, K=mv²/2, and the derived relation K=p²/(2m). The latter follows by substituting v=p/m into kinetic energy; it is not an independent law. Every inversion uses positive magnitudes, for one constant-mass particle at one instant and in one inertial frame. The engine searches acyclic routes in this small graph and retains the shortest route for each distinct observed-input set. It is not a general symbolic algebra system.

The target is always excluded from observations. Empty/invalid data is not imputed. Answers that differ by more than 0.5% of the maximum trigger an inconsistency warning; this is a display tolerance, not an experimental uncertainty model. Shared input sets or shared measurements do not constitute independent empirical confirmation.

Optional WebMCP tools read or configure exactly the same local page state. They never submit GitHub reports. The site collects no data and has no analytics. Google Fonts may receive normal font requests; system fonts provide a fallback.

## Next scientific milestone

Connect a separately versioned GNN inference endpoint or validated browser export; show its prediction next to, not disguised as, symbolic calculations. Benchmark identical observation masks; include a complete symbolic baseline, repeated training runs, physical context annotations, and independently measured data. Preserve published artifacts unchanged.

## Licensing

Code: MIT, copyright 2026 Md Minnatullah. The linked paper and research artifacts retain their own licenses.
