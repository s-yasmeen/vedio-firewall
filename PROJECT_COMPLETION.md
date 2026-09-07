# TAPF-MIN Project Completion Status

## Product thesis
TAPF-MIN is a representation-first biometric release firewall for telemedicine. Raw biometric video is processed locally where possible. The system selects the minimum authorized representation, verifies empirical privacy/utility evidence, and fails closed when configured bounds are not met.

## Implemented system layers

### 1. On-device / edge representation layer
- Motion features for movement tasks.
- Physiological-signal research proxy for rPPG experiments.
- Action-unit adapter for expression tasks.
- Cancelable protected biometric template path for authentication.
- Protected-video path intentionally refuses release unless separately verified.

### 2. Privacy–utility research layer
- Static TAPF baselines.
- Motion-representation benchmark.
- Minimum-disclosure spectrum.
- Linear identity-suppression baseline.
- TAPF-MIN v2.1 feature-level adversarial/disentanglement prototype.
- Actor-disjoint task evaluation.
- Independent post-hoc identity attackers.
- Confidence intervals and negative controls.

### 3. Release controller
- Conservative evidence bounds.
- Privacy upper-bound threshold.
- Utility lower-bound threshold.
- Temporal-risk upper-bound threshold.
- Latency bound.
- Authorized task/representation mapping.
- Fail-closed BLOCK decision.

### 4. Privacy–utility attestation
- Request ID.
- Evidence SHA-256 digest.
- Task and representation.
- Release/BLOCK decision and reason.
- Thresholds and selected operating point.
- Raw-biometric-transmitted flag.
- Optional HMAC signing for prototype deployment.

### 5. Application / PWA
- Installable TAPF-MIN Edge PWA.
- Local browser camera demonstration using getUserMedia.
- No camera-frame upload path in the interface.
- Release-gate UI connected to /v2/release/evaluate.
- Session attestation history.
- Service health/readiness indicators.

### 6. Deployment hardening
- Docker image support.
- Kubernetes manifest.
- API-key authentication support.
- Signing-secret readiness checks.
- Health and readiness endpoints.
- Security/deployment documentation.

## Scientific boundaries
- Current v2.1 benchmark is a feature-level adversarial/disentanglement prototype using the validated motion descriptor; it is not yet a clinically validated raw-video deep encoder.
- Passing tested attackers is empirical evidence under the tested threat model, not proof of anonymity.
- CREMA-D emotion recognition is a controlled utility benchmark, not a clinical diagnostic validation.
- Protected-video release remains blocked unless a separately validated protected-video pathway satisfies the release criteria.

## Remaining research upgrades after competition prototype
1. Raw-video MobileNetV3/EfficientNet + temporal encoder.
2. Independent external dataset validation.
3. Neural/open-set/linkage attackers.
4. Temporal accumulation attack across repeated releases.
5. Healthcare-specific task validation.
6. Edge latency, energy and memory profiling on a target mobile/embedded device.
7. Fairness/demographic subgroup analysis.

## Competition-ready system claim
TAPF-MIN demonstrates a fail-closed architecture that asks what biometric information must leave the patient device, releases only an authorized minimum representation, evaluates residual identity exposure and task utility, and records the resulting decision.

Do not claim formal anonymity, clinical validation, or regulatory compliance from the current prototype.
