# TAPF-MIN Security Model

## Security objective
TAPF-MIN minimizes biometric disclosure and blocks release unless measured privacy, task utility, temporal leakage, representation authorization, freshness, and latency criteria are satisfied. It is an empirical control system, not a proof of anonymity.

## Trust boundary
- Raw biometric video remains on the patient/edge device.
- The release-decision API accepts measured evidence only; it has no raw-video ingestion endpoint.
- Authentication uses a protected/cancelable template path, separate from care-task representations.

## Threats explicitly considered
1. Frame-level face recognition after protection.
2. Sequence-level identity aggregation across video frames.
3. Weak-attacker optimism; deployment uses worst-case evidence across independent attackers.
4. Stale privacy/utility measurements.
5. Unauthorized representation release for a task.
6. Face-localization failure in a release-critical transform.
7. Evidence tampering; release attestations can be HMAC signed.
8. Configuration drift; `/readyz` requires release API authentication and attestation signing.
9. Protected-template compromise; cancelable templates support empirical key rotation/revocation testing.
10. Latency overload that could invalidate real-time operating assumptions.

## Fail-closed conditions
TAPF-MIN must BLOCK or remain local-only when any required evaluator is unavailable, evidence is stale/incomplete, confidence bounds fail, no representation is authorized, localization fails, a modern attacker remains above threshold, or latency exceeds policy.

## Known external requirements
The repository does not by itself provide regulated clinical validation, medical-device authorization, production key management/HSM, enterprise identity management, a distributed replay store, independent penetration testing, or legal/privacy compliance certification. Those controls must be supplied by the deployment environment and validated independently.

## Secrets
Never commit `TAPF_API_KEY` or `TAPF_ATTESTATION_SECRET`. Use a secret manager in production. Rotate API and attestation credentials and record the attestation key ID.

## Reporting
Security findings should be reported privately to the repository owner before public disclosure when practical.
