# TAPF-MIN Deployment Runbook

## Status
This repository is a deployment-hardened research system, not a regulated clinical product.

## Architecture
1. Raw video is processed on the patient/edge device.
2. A task-conditioned representation is selected. Prefer derived representations such as action units, motion features, physiological signals, or cancelable templates over human-viewable video.
3. Independent identity attackers and the task evaluator produce measured evidence.
4. Confidence bounds, temporal risk, freshness, and latency are checked.
5. The release service returns RELEASE or BLOCK and emits a signed Privacy-Utility Attestation.

## Required production configuration
- `TAPF_API_KEY`: bearer credential for `/v2/release/evaluate`.
- `TAPF_ATTESTATION_SECRET`: signing secret. Replace local HMAC secret management with an HSM/KMS-backed signing service for production.
- `TAPF_ATTESTATION_KEY_ID`: auditable key identifier.

`/healthz` is liveness. `/readyz` is deployment readiness and returns 503 if authentication/signing are not configured.

## Recommended runtime
- Edge: ONNX Runtime adapter (`requirements-edge.txt`) with device-specific execution provider benchmarked on target hardware.
- Release service: containerized FastAPI service. It does not accept raw biometric frames.
- TLS, WAF/API gateway, rate limits, centralized audit logging, KMS/HSM, and distributed replay/idempotency storage belong in the hosting platform.

## Release evidence
Production should use `/v2/release/evaluate`, not the legacy point-estimate endpoint. Each operating point must contain identity risk and its upper confidence bound, utility and its lower confidence bound, temporal risk and its upper bound, p95-equivalent latency evidence, evaluator identity, and sample count.

The controller releases only when all conservative constraints pass. Evidence must be freshly measured and generated under the deployed model versions/threat model.

## Current empirical warning
The present Static TAPF visual transform is not deployment-eligible: CREMA-D FaceNet experiments show sequence identity remains highly separable after the current transform. The system must therefore prefer minimum-disclosure derived representations and BLOCK protected-video release until a transform passes the configured attacker ensemble.

## Pre-production acceptance gates
- Core and deployment CI green.
- Container starts as non-root and `/readyz` is green with secrets configured.
- Independent attacker ensemble includes at least two modern face recognizers.
- Subject-disjoint task utility validated on the intended task and population.
- Confidence bounds calibrated on held-out validation data.
- Target-device p50/p95/p99 latency, memory, power, thermal, and FPS benchmark completed.
- Cancelable authentication EER/TAR@FAR, revocability, unlinkability, and inversion-resistance protocol completed.
- External security review/penetration test completed.
- Clinical validation and regulatory/privacy approvals completed when used for healthcare decision-making.

## Deployment decision vocabulary
- `RELEASE`: all configured empirical constraints satisfied.
- `BLOCK`: one or more required constraints failed.
- `UNVERIFIED`: allowed only in non-production research policy; production must remain fail-closed.
