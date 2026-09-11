# TAPF-MIN — Falling Walls Lab executable demo

## Purpose

This demo presents the TAPF-MIN minimum-disclosure release architecture without fabricating FER performance. It uses the repository's real randomized-response mechanism and the frozen reported research measurements.

## Preferred Falling Walls launch — standalone Windows executable

Download the GitHub Actions artifact named:

`TAPF-MIN-Falling-Walls-Windows`

Extract it, then double-click:

`TAPF-MIN-Falling-Walls.exe`

No separate Python installation is required on the presentation laptop. The executable starts a local Streamlit server on `127.0.0.1` and opens the demo in the default browser.

For presentation safety, test the executable once on the exact laptop you will take to the venue and keep the extracted folder locally rather than launching it from a cloud-synced folder.

## Source-mode fallback on Windows

If needed, double-click:

`demo/run_falling_walls_demo.bat`

The first run installs the small presentation dependencies and opens the Streamlit app in a browser.

## macOS/Linux source-mode launch

```bash
bash demo/run_falling_walls_demo.sh
```

## Presentation flow

1. Open **Live Firewall**.
2. Explain that the six-class posterior is produced locally. In this presentation build it is an explicitly labelled demonstration input because no trained FER checkpoint is committed to this branch yet.
3. Click **Request protected release**.
4. Show that only a protected categorical task label leaves the private boundary.
5. Repeat the release until the session privacy budget is exhausted. The system then returns **BLOCK**.
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

## Build verification

The GitHub Actions Windows job must:

1. build `TAPF-MIN-Falling-Walls.exe` with PyInstaller;
2. verify that the executable exists;
3. launch the executable on Windows;
4. verify the local interface returns HTTP 200;
5. upload the tested executable as the `TAPF-MIN-Falling-Walls-Windows` artifact.

## What can be said on stage

> Instead of sending a patient's face to an AI system, TAPF-MIN keeps the biometric video local and releases only the minimum task information under a measurable privacy budget.

Do not say that the current CREMA-D experiment proves anonymity, clinical validity, or final FER performance. The value of the demo is the architecture and measurable release control, not a fabricated benchmark win.
