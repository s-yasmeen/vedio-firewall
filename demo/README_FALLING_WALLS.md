# TAPF-MIN — Falling Walls Lab executable demo

## Purpose

This package presents the TAPF-MIN minimum-disclosure release architecture without fabricating FER performance. The standalone executable uses the repository's real k-ary randomized-response mechanism and the frozen reported research measurements.

## Preferred Falling Walls launch — standalone Windows executable

Download the GitHub Actions artifact named:

`TAPF-MIN-Falling-Walls-Windows`

Extract it, then double-click:

`TAPF-MIN-Falling-Walls.exe`

No separate Python installation is required on the presentation laptop. The executable starts a self-contained local web server bound to `127.0.0.1` and opens the demo in the default browser. It does not require Streamlit at runtime.

For presentation safety, test the executable once on the exact laptop you will take to the venue and keep it on the local disk rather than launching it from a cloud-synced folder.

## App-development prototype interface

The executable exposes a small local HTTP interface that can later be replaced by a desktop/mobile/web frontend:

- `GET /healthz` — runtime health check
- `GET /api/evidence` — frozen research evidence displayed by the demo
- `POST /api/release` — exercises the real TAPF randomized-response categorical release
- `POST /api/reset` — resets the presentation-session demonstration budget

The executable's in-memory epsilon counter is a **presentation-session controller**, not the production persistent privacy ledger. Production integration should use the repository's server-side release authority and persistent accounting layer.

## Source-mode development UI

The existing Streamlit UI remains available for source-mode development:

Windows:

`demo/run_falling_walls_demo.bat`

macOS/Linux:

```bash
bash demo/run_falling_walls_demo.sh
```

## Presentation flow

1. Open **Live Firewall**.
2. Explain that the six-class posterior is produced locally. In this presentation build it is explicitly labelled as a demonstration input because no trained final FER checkpoint is committed to the branch yet.
3. Click **Request protected release**.
4. Show that only a protected categorical task label leaves the private boundary.
5. Repeat the release until the presentation-session privacy budget is exhausted. The system returns **BLOCK**.
6. Open **Measured Evidence** and show the frozen v2.1 result and v3 development diagnostic exactly as measured.
7. Open **Architecture** and explain that raw video, face crops and task latents are not authorized release objects.

## Evidence included in the app

### TAPF-MIN v2.1 frozen CREMA-D validation

- 144 clips / 24 actors
- emotion macro-F1: 0.230
- macro-F1 95% CI: 0.165–0.297
- clip identity AUC: 0.573
- identity AUC 95% CI: 0.474–0.665
- repeated-release identity AUC: 0.693
- decision: **BLOCK**

### v3 deep-FER development run

- 144 clips / 24 actors
- local FER accuracy: 0.1597
- local macro-F1: 0.1505
- macro-F1 95% CI: 0.091–0.212
- identity-posterior attack ≈0.998 effective AUC, explicitly labelled as confounded by fold/checkpoint signatures
- decision: **BLOCK / diagnostic only**

### v3.1 hardening

- engineering CI: PASS
- face-focused preprocessing
- bounded box reuse
- RGB + motion fusion
- temporal GRU/attention
- delayed identity adversary
- fail-closed prevalidation gate
- final FER/privacy benchmark: **not yet claimed**

## Final executable verification gate

The GitHub Actions Windows job must pass all of the following before the file is treated as validated:

1. compile the source and verify frozen evidence;
2. build `TAPF-MIN-Falling-Walls.exe` with PyInstaller;
3. verify the executable exists;
4. launch the executable on a Windows runner;
5. verify `GET /healthz` responds successfully;
6. submit a protected release and require **RELEASE** with a 3-bit categorical payload and no raw-video/task-latent release;
7. repeat against an exhausted budget and require fail-closed **BLOCK**;
8. upload the exact tested executable as the `TAPF-MIN-Falling-Walls-Windows` artifact.

This is executable/function validation, not final scientific validation of FER accuracy or clinical performance.

## What can be said on stage

> Instead of sending a patient's face to an AI system, TAPF-MIN keeps the biometric video local and releases only the minimum task information under a measurable privacy budget.

Do not say that the current CREMA-D experiment proves anonymity, clinical validity, or final FER performance. The strength of the executable is the architecture and measurable release control, not a fabricated benchmark win.
