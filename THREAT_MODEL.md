# TAPF-MIN Threat Model — Frozen Research/Demo Scope

## Protected asset
Patient biometric identity and biometric-derived information released during a telemedicine session.

## Trust boundary
Raw biometric capture and task-feature extraction are intended to occur on the patient device/edge. The server-side release service accepts evidence and release metadata, not raw biometric frames.

## Adversaries considered
1. **Single-release identity attacker** — infers patient identity from one released representation.
2. **Repeated-release attacker** — aggregates multiple releases from the same patient/session to infer identity.
3. **Recipient misuse** — a recipient attempts to use a release for a task/purpose different from the authorized contract.
4. **Replay attacker** — reuses an expired or previously consumed task authorization/consent artifact.
5. **Representation substitution** — requests a representation not authorized for the clinical task.
6. **Stale-evidence attacker** — attempts release using evidence outside the accepted freshness window.
7. **Tampering attacker** — changes task, purpose, recipient, representation, evidence, or policy metadata after evaluation.
8. **Input degradation** — sensor failure or low-quality input creates unreliable evidence; the system must degrade explicitly or block.

## Primary measured privacy criterion
Identity privacy is evaluated with independent post-hoc attackers using chance-centered effective AUC:

`AUC_eff = 0.5 + |AUC - 0.5|`

Prospective engineering release criterion:
- clip identity effective-AUC upper bound <= 0.55;
- every independent attacker effective AUC <= 0.55;
- repeated-release effective AUC <= 0.55;
- task Macro-F1 lower 95% confidence bound >= 0.20;
- reliability and authorization checks pass;
- evidence is fresh and complete.

## Fail-closed behavior
TAPF-MIN blocks release when any required authorization, evidence, privacy, repeated-release, utility, recipient-binding, freshness, or reliability condition fails. Unknown tasks have no default fallback representation.

## Out of scope / not claimed
This research prototype does **not** claim formal anonymity, differential privacy, universal resistance to all future attackers, clinical diagnostic efficacy, medical-device safety, or regulatory compliance. Real clinical validation remains a separate evidence requirement.

## Validation boundary
Public affective/biometric datasets and synthetic clinical-like stress tests provide technical evidence only. Clinical claims require appropriately governed clinical cohorts and task-specific validation.
