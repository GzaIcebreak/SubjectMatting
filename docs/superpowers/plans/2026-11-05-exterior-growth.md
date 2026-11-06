# Exterior growth protection implementation plan

Goal: reject the unsafe whole-candidate substitutions in05/19 without case/class checks, preserve17/20/43/41, replay47 through the packaged app.
Architecture: A defines a conservative outer support envelope. Fill A's enclosed holes ONLY in a temporary decision mask, allow a narrow scale-normalized boundary band, measure B's added alpha mass beyond that support. Significant expansion vetoes B independently of the cloud opinion. Actual candidate pixels remain unchanged. Existing target, continuity and optical vetoes remain.
Spec: user authorized direct implementation of a general no-training fix and47-case regression.
Alternatives considered: stronger prompts remain vulnerable to the observed hallucinations; image edge/colour matching was probed but confuses low-contrast real parts and textured background. Choose explicit conservative geometry, with acknowledged inability to recover an entirely missing external part. No per-image switches or training.
Constants: alpha support>=32/255; working long side<=1024, any/all foreground pooling preserves local support and exterior connectivity; average alpha; hole floodfill4-connected; boundary band max(2,round(1% shortest working side)), Chebyshev distance; expansion mass>4 pixels and>2% original alpha mass. This is a conservative tolerance policy, not a calibrated probability.

- [x] Add failing behavioral tests: attached background inside a loose target box despite approving review; interior restoration and narrow boundary corrections allowed; scaling/translation/aspect changes; optical/continuity protection retained. Test coarse downsampling separately.
- [x] Implement bounded topology helper in MattingQ/ExteriorGrowth.swift; expose audit metrics and independent veto in HybridSelection; worker forwards it in both decision paths.
- [x] Replay frozen47 A/B with actual Swift selector via test-only harness, document all changed choices (including missed improvements), then run full Swift tests and rebuild/sign app.
- [x] Fresh packaged-worker47 replay into outputs/hybrid_1105/final_run, priority05/19/17/20/43/41. New report compares old hybrid, same-runA/B, final. Verify delivery hashes, selection and protected outputs. Preserve prior artifacts. No push or SAM2 changes.

Review fixes: add narrow open-channel tests before fixing pooling; use all-foreground for flood barriers and any-foreground only for local support. Frozen replay asserts reviewed choices for every one of47 cases.15 and27 conservative rejections explicitly disclosed.

Final: Swift46 pass; full packaged47 attempted,46 exported.19 auto retry still empty; explicit road control exported with exterior veto. No fallback detection product changes. Protected17/20/43 equal previous hybrid;41 equal earlier originalApp. Primary output/hash/report checks passed. Known15/27 conservative tradeoff and26 cloud variance documented.
