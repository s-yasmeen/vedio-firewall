# TAPF-MIN Project Completion Status — v2.2 Release Candidate

## Product thesis
TAPF-MIN is a representation-first biometric release firewall for telemedicine. Raw biometric video is processed locally where possible. The system selects the minimum authorized representation, verifies empirical privacy/utility evidence, and fails closed when configured bounds are not met.

## Current status
The engineering prototype is release-candidate ready for demonstration/research use. The privacy protocol, runtime gate, repeated-release controls, application shell, deployment hardening, attestation, and reproducible validation workflows are implemented. Real clinical validation remains an external evidence dependency and is not claimed as complete.

## Implemented system layers

### 1. On-device / edge representation layer
- Motion features for movement tasks.
- Physiological-signal research proxy for rPPG experiments.
- Action-unit adapter for expression tasks.
- Cancelable protected biometric template path for authentication.
- Protected-video path intentionally refuses release unless separately verified.
- No raw-camera upload path in the demonstration interface.

### 2. Privacy–utility research layer
- Static TAPF baselines.
- Motion-representation benchmark.
- Minimum-disclosure spectrum.
- Linear identity-suppression baseline.
- TAPF-MIN v2.1 adversarial/disentanglement prototype.
- TAPF-MIN v2.2 privacy-amplification benchmark.
- Actor/patient-disjoint task evaluation.
- Independent post-hoc identity attackers.
- Repeated-release aggregation attack.
- Confidence intervals and negative controls.
- Chance-centered identity AUC so AUC < 0.5 is not incorrectly treated as safe.
- Synthetic clinical-like stress benchmark for controlled pre-clinical validation only.

### 3. Frozen v2.2 empirical release gate
A release is eligible only when all of the following pass:
- conservative clip-level effective identity AUC <= 0.55,
- repeated-release effective identity AUC <= 0.55,
- every supplied independent identity attacker has effective AUC <= 0.55,
- lower 95% confidence bound of task Macro-F1 >= 0.20,
- required evidence is complete,
- representation is authorized for the requested task,
- deployment latency bound passes where applicable.

The gate is fail-closed. Thresholds must not be relaxed after observing results.

### 4. Runtime release controller
- Dedicated `tapf.deployment_v22` adapter uses the same chance-centered gate as the research benchmark.
- Generic controller defaults now use v2.2-compatible identity/repeated-release advantage semantics.
- Invalid/NaN evidence fails closed.
- Unauthorized task/representation combinations are blocked.
- Repeated disclosure requires explicit aggregate-risk evidence.

### 5. Privacy–utility attestation
- Request ID.
- Evidence SHA-256 digest.
- Task and representation.
- RELEASE/BLOCK decision and reason.
- Thresholds and selected operating point.
- Raw-biometric-transmitted flag.
- Optional HMAC signing for prototype deployment.

### 6. Application / PWA
- Installable TAPF-MIN Edge PWA.
- Local browser camera demonstration using getUserMedia.
- No camera-frame upload path in the interface.
- Release-gate UI connected to the release service.
- Session attestation history.
- Service health/readiness indicators.

### 7. Deployment hardening
- Docker image support.
- Kubernetes manifest.
- API-key authentication support.
- Signing-secret readiness checks.
- Health and readiness endpoints.
- Security/deployment documentation.
- Dependency security audit workflow.
- Smoke/deployment tests.

## Validation ladder

### Level A — real public research datasets
Used for empirical privacy/utility validation and external replication where task labels permit it. CREMA-D is the current primary benchmark. Additional public external datasets such as RAVDESS/LFW should be used according to the labels and threat models they genuinely support.

### Level B — synthetic clinical-like stress validation
Implemented in `experiments/run_synthetic_clinical_stress.py`.
This controlled benchmark generates longitudinal synthetic patient sessions with separate identity nuisance and clinical-like movement-severity factors, optional identity/task confounding, patient-disjoint task evaluation, independent identity attacks, and repeated-release attacks.

**Claim boundary:** this is not clinical validation, diagnostic validation, patient-safety evidence, or evidence of regulatory compliance. It is a controlled stress/ablation benchmark that can expose privacy-controller failures before real clinical data are available.

### Level C — real clinical validation
Pending access/collaboration. Required before any claim that TAPF-MIN preserves clinically diagnostic information or is suitable for clinical deployment.

## Known evidence boundary
- The current learned benchmark is still feature-level rather than a fully clinically validated raw-video temporal encoder.
- Passing tested attackers is empirical evidence under the tested threat model, not proof of anonymity.
- Public affect/emotion datasets are not substitutes for clinical diagnostic cohorts.
- Synthetic clinical-like data do not substitute for real patients.
- Regulatory compliance is not claimed.

## Remaining external research dependencies
1. Independent external real-dataset validation beyond CREMA-D.
2. Real clinical/telemedicine dataset validation with medically meaningful endpoints.
3. Clinical expert review of preserved signals and acceptable utility thresholds.
4. Target-device latency, energy and memory profiling on representative mobile hardware.
5. Fairness/demographic subgroup validation on datasets that support those analyses.

These are validation dependencies, not missing core product components.

## Release-candidate claim
TAPF-MIN v2.2 demonstrates a working fail-closed biometric release architecture that asks what biometric information must leave the patient device, releases only an authorized minimum representation, measures residual identity exposure at clip and repeated-release levels, verifies task utility, blocks unsafe operating points, and records the decision.

Do not claim formal anonymity, completed clinical validation, diagnostic efficacy, medical-device readiness, or regulatory compliance from the current prototype.
