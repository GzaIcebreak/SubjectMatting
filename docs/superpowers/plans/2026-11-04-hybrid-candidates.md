# Hybrid candidates implementation plan

Goal: implement the user's approved no-training two-route selection in the Q macOS app and replay all 47 report inputs, explicitly checking smoke/kettle/perfume/17/20.
Architecture: original Q pipeline remains candidate A. A lazy BiRefNet session produces continuous detail alpha and a traditional-trimap ViTMatte candidate B. The worker computes candidate disagreement/completeness, obtains a structured semantic review (original, two candidate composites), defaults to A on ambiguity/error, and only accepts B when review and local integrity checks agree. Thin-line preservation uses bounded local alpha differences within B's uncertain boundaries; no class/file-name case switches.
Tech: existing C++ ONNX Runtime, Swift MattingQ and worker, Python artifact runner.
Spec: approved design in chat (whole-candidate choice, limited local refinement; no training).

Constraints: Q branch only; preserve existing edits; SAM2 unchanged; no key logging; no pushes; frozen original outputs retained. No case IDs/labels used for routing. Model confidence is not truth. Do not promise improvement from a successful execution alone.

Review focus: added unrelated subjects / reflection; missing base or whole body; thin branches lost by matting; transparent materials with false holes; reviewer failure and malformed replies.

- [x] C++ additive `smq_detail` API producing direct and refined RGBA at source dimensions. Use BiRefNet_lite at1024; same pre/postprocess as offline reference. Lazy load optional model, bounded CPU threads. Test numerical parity and errors.
- [x] Swift selector types/parser and tests: default A for uncertain votes, missing parts, transparency risk; accept B only with affirmative complete/target-consistent verdict; local thin-edge protection never fills definite B-background. Original alpha cap applies after selection.
- [x] Bind C API, integrate two-route semantic review with cropped disagreement evidence and logging. Audit output includes A/B/final, detection, metrics, decision/reason. No mandatory semantic cloud judge if candidates effectively identical.
- [x] Package BiRefNet weights, build and ad-hoc sign app. Run Swift/C++ checks.
- [x] Run packaged worker on all47 copied inputs; save actual same-run A/B/final, compare against previous app outputs. Highlight protected cases and failures without cherry-picking. Generate HTML review report and provenance.

Result: 47/47 exported; A17/B30. Protected17/20 exactly match previous App; all5 protected cases keep same-run A. Full provenance checks passed. Known failures05/19 and unresolved dense holes documented in `docs/diagnostics/2026-11-04-hybrid-candidates.md`; no universal quality claim.
