# TAPF-MIN

**Verified Minimum-Disclosure Biometric Release for Telemedicine Video**

TAPF-MIN is a research prototype for controlling what biometric information is allowed to leave a patient device. It is not another fixed anonymizer. The core research question is:

> What is the minimum task-sufficient representation that can be released while measured identity exposure remains below a predefined risk threshold?

## Core mechanisms

TAPF-MIN implements six first-class mechanisms:

1. **Minimum disclosure** — select the smallest authorized task representation rather than automatically transmitting a face/video stream.
2. **Task-conditioned release** — preservation targets depend on the authorized downstream task.
3. **Attacker-verified privacy** — release depends on measured identity risk, not visual appearance alone.
4. **Temporal privacy accounting** — sequence-level evidence is monitored because identity leakage can accumulate across repeated releases.
5. **Fail-closed behavior** — if no operating point satisfies the release policy, the system blocks release.
6. **Privacy–utility attestation** — release decisions can emit a machine-readable record of the evidence, thresholds and decision.

## Integrated prototype status

The current competition prototype includes:

- Edge representation adapters for motion, physiological-signal research proxy, action-unit integration, and cancelable authentication templates.
- Conservative `/v2/release/evaluate` API using privacy upper bounds, utility lower bounds, temporal upper bounds, latency limits and authorized task/representation mappings.
- Optional API authentication and signed prototype attestations.
- Installable TAPF-MIN Edge PWA.
- A browser local-camera demonstration using `getUserMedia`; the interface contains no camera-frame upload path.
- Docker/Kubernetes deployment assets and health/readiness endpoints.
- Real-data CREMA-D privacy–utility benchmarks including minimum-disclosure and v2.1 adversarial/disentanglement experiments.

See `PROJECT_COMPLETION.md` for the implementation boundary and remaining research upgrades.

## Scientific guardrails

- A visually altered face is not assumed to be private.
- Passing the tested attacker ensemble is empirical evidence under that threat model, not proof of anonymity.
- CREMA-D emotion recognition is a controlled utility benchmark, not clinical diagnostic validation.
- The current v2.1 model is a feature-level adversarial/disentanglement prototype using the validated temporal motion descriptor; a final raw-video MobileNet/temporal encoder remains a research upgrade.
- Protected-video release remains fail-closed unless separately validated.
- Simulation outputs are not research results.

## Benchmark datasets

- **CREMA-D** — primary video benchmark: actor identity + temporal facial emotion task in the same clips.
- **LFW** — external identity/privacy benchmark.
- **FER+ / FER2013** — external expression-utility validation.

Do not commit full biometric datasets. Preserve dataset licenses and access conditions.

## Architecture

```text
PATIENT DEVICE / EDGE

Camera / video
    |
    v
Authorized task
    |
    v
Minimum task representation
    |
    +------> independent identity-risk evaluation
    |
    +------> task-utility evaluation
    |
    +------> temporal-risk evaluation
    |
    v
Conservative Privacy–Utility Gate
        /              \
     PASS              FAIL
      |                  |
   RELEASE            BLOCK
      |
      v
Privacy–Utility Attestation
```

The release-decision service itself accepts **measured evidence**, not raw biometric frames.

## Local prototype

```bash
pip install -r requirements-deploy.txt
uvicorn service.app:app --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/app/`.

For deployment-like readiness checks, configure:

```bash
export TAPF_API_KEY='replace-me'
export TAPF_ATTESTATION_SECRET='replace-me'
```

Then run:

```bash
pytest -q
python experiments/run_simulation.py
```

## Research evaluation path

Current experiments include:

- conventional visual-transform baselines;
- CREMA-D minimum-representation benchmark;
- privacy–utility disclosure spectrum;
- cross-fitted linear identity-suppression baseline;
- TAPF-MIN v2.1 adversarial/disentanglement feature-level benchmark;
- independent post-hoc identity attacks and confidence intervals.

Next research upgrades are external-dataset validation, raw-video temporal encoding, neural/open-set/linkage attackers, repeated-release temporal attacks, device profiling, fairness evaluation and healthcare-specific task validation.

**Only final measured results should appear in a publication or competition slide.**
