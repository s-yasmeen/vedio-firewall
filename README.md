# TAPF-MIN

**Verified Minimum-Disclosure Biometric Release for Telemedicine Video**

TAPF-MIN is a research prototype for controlling what biometric information is allowed to leave a patient device. It is not another fixed anonymizer. The core research question is:

> What is the minimum task-sufficient representation that can be released while measured identity exposure remains below a predefined risk threshold?

## Novel research core

TAPF-MIN implements six first-class mechanisms:

1. **Minimum disclosure** — search for the weakest transformation that satisfies privacy and utility constraints.
2. **Task-conditioned release** — preservation targets depend on the authorized downstream task.
3. **Attacker-verified privacy** — release depends on measured identity risk, not visual appearance alone.
4. **Temporal privacy accounting** — sequence-level evidence is monitored because identity leakage can accumulate across video frames.
5. **Fail-closed behavior** — if no operating point satisfies the release policy, the protected stream is blocked.
6. **Privacy–utility attestation** — every release decision can emit a machine-readable record of the measured operating point and policy thresholds.

## Important scientific guardrail

The repository initially includes a deterministic simulation harness so the controller can be executed and tested without biometric data. Simulation outputs are **not research results**. Publication claims require replacing the simulation adapters with measured identity attackers and downstream task models.

## Recommended real-data benchmark

- **CREMA-D** — primary video benchmark: actor identity + temporal facial emotion task in the same clips.
- **LFW** — external identity/privacy benchmark.
- **FER+ / FER2013** — external expression-utility validation.

Do not commit the full biometric datasets to this repository. Add dataset download/preparation scripts or documented manual-access steps and preserve license conditions.

## Architecture

```text
Patient video
    |
    v
Task policy ---> Candidate representation / transform(alpha)
    |                         |
    |                         v
    |                 Identity attacker(s)
    |                         |
    |                         v
    |                 Temporal risk account
    |                         |
    |                         v
    +-----------------> Utility evaluator
                              |
                              v
                    Privacy–Utility Gate
                         /          \
                      PASS          FAIL
                       |              |
                    RELEASE       ADAPT/BLOCK
                       |
                       v
                  Attestation
```

## Quick start

```bash
pip install -r requirements.txt
python experiments/run_simulation.py
pytest -q
```

## Publication path

Replace the simulation adapters with:

- ArcFace + FaceNet + an unseen independent attacker for identity leakage.
- CREMA-D subject-disjoint emotion classifier for task utility.
- Sequence-level identity aggregation for temporal leakage.
- Original, blur, pixelation, static TAPF, adaptive TAPF, and TAPF-MIN as baselines.

Only measured results should appear in a paper or competition slide.
