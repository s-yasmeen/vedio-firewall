# TAPF-MIN v3.1 Pre-validation Red-Team Audit

This audit exists to prevent a visually improved FER architecture from being mistaken for a scientifically improved privacy system. The system remains BLOCK until measured evidence passes the frozen gates.

## 1. Input/preprocessing leakage

### Threats
- Whole-frame or center-crop fallbacks can expose background, clothing, camera geometry, lighting and session identity.
- Face detector failure rate can itself become an identity/session signal.
- Eye alignment can fail systematically for particular actors, poses, eyewear or illumination.
- Reused bounding boxes may drift onto background or clothing.
- Preprocessing metadata (detection/alignment/reuse flags) can leak if transmitted.

### Tightening
- First-frame detector failure now produces a neutral crop, not scene content.
- Short detector dropouts may reuse the previous trusted box but are explicitly counted.
- Invalid frames are masked out of temporal attention and all-invalid training sequences are skipped.
- Detection, alignment, valid-face and box-reuse rates are measured internally only.
- Prevalidation requires adequate face-valid/detection coverage and limits actor-dependent detection disparity.

## 2. Motion-stream leakage

### Threats
- Frame differences can preserve identity through facial geometry, skin texture edges, head shape and stable alignment artifacts.
- Motion can encode camera shake, background motion or session-specific compression.
- A zero first residual creates a deterministic position marker.

### Controls / remaining evidence
- Motion operates only after face-focused preprocessing.
- Motion representation is never an authorized release object.
- Held-out verification/linkage attacks must be run on the final released representation.
- Future optional diagnostic: probe RGB, motion and fused internal latents separately to understand where identity information concentrates. These internal probes are empirical diagnostics, not formal privacy guarantees.

## 3. Identity-adversary failure modes

### Threats
- Strong GRL too early can destroy FER utility before task features form.
- Seen-identity adversarial success does not prove unseen-identity privacy.
- Identity adversary may learn dataset shortcuts while leaving biometric linkage intact.

### Tightening
- GRL is delayed and smoothly ramped.
- Identity loss weight and GRL maximum are bounded/configurable.
- Final privacy evidence must come from held-out identities and one frozen FER model.
- PASS requires clip identification, repeated-release, verification and linkage upper confidence bounds within policy.

## 4. Training/evaluation leakage

### Threats
- Actor overlap between train/test invalidates utility and privacy estimates.
- Selecting an architecture on OOF folds and reporting the same OOF score as final evidence creates selection bias.
- Different checkpoints per privacy-audit actor can create fold/model fingerprints.
- Clip-wise bootstrap treats correlated clips as independent.
- Hyperparameter tuning on the final cohort silently contaminates the test set.

### Tightening
- Development ablation is actor-disjoint and explicitly labeled architecture-selection only.
- Selected architecture must be frozen before a separate held-out actor audit.
- Privacy audit must transform all audit identities with one frozen checkpoint.
- Utility uncertainty should be actor-clustered; privacy uncertainty should be hierarchical/identity-aware where feasible.
- Any change after final-cohort inspection creates a new protocol/version and requires a fresh final cohort where possible.

## 5. Formal privacy / release leakage

### Threats
- Good empirical identity AUC does not prove differential privacy.
- Differential privacy of the task release does not imply anonymity or protect timing/logging side channels.
- Repeated releases compose and can exhaust the budget.
- Raw video, task latents or preprocessing metadata would bypass the minimum-disclosure boundary.

### Tightening already present
- Formal DP mechanism is applied only at the release layer.
- Persistent scope-bound accounting survives restarts.
- Replay-safe release IDs prevent duplicate spend ambiguity.
- Frozen server-side evidence binds model/protocol/mechanism.
- Raw video and task latent release are explicit BLOCK conditions.

## 6. Side-channel leakage still outside the theorem

Must be audited before production claims:
- response timing and BLOCK/RELEASE timing,
- payload length/type beyond task payload bits,
- HTTP headers, session identifiers and network metadata,
- logs, crash dumps, caches and telemetry,
- model-loading/version errors,
- OS/device compromise,
- debug endpoints and saved intermediate tensors.

These are deployment-security concerns and are not covered by current DP guarantees.

## 7. Pre-validation PASS criteria

The code-level `PrevalidationPolicy` currently requires all evidence to be present and fail-closes otherwise. Development gates include:
- macro-F1 materially above the frozen v3 baseline,
- macro-F1 lower confidence bound >= 0.55,
- valid face rate >= 0.90,
- face detection rate >= 0.80,
- actor detection-rate gap <= 0.15,
- identity/repeated/verification/linkage AUC upper confidence bounds <= 0.60,
- actor overlap = 0,
- one frozen model for the privacy audit,
- formal DP evidence present,
- raw video released = false,
- task latent released = false.

These are development progression gates, not clinical or universal safety guarantees. Stronger final-publication thresholds may be frozen later before the final evaluation.

## 8. Claims that remain prohibited until evidence exists

Do not claim:
- FER is fixed,
- identity privacy is solved,
- anonymity,
- clinical validity,
- generalization to telemedicine,
- superiority over published baselines,
- production-grade side-channel security.

Permitted statement before final evaluation: the architecture has been hardened to reduce known leakage paths and is subject to a fail-closed evidence gate.
