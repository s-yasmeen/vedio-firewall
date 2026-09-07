# TAPF-MIN Competition Demo Runbook

## Objective
Demonstrate the project as an integrated privacy-release system without overstating current research maturity.

## Demo sequence

1. Open the TAPF-MIN Edge PWA.
2. Select **Device Demo**.
3. Start the local camera.
4. Point out the **LOCAL ONLY** indicator and explain that this UI does not upload camera frames.
5. Stop the camera.
6. Open **Release Gate**.
7. Select an authorized healthcare task and representation.
8. Enter measured privacy, utility, temporal and latency evidence from a validated experiment.
9. Run the gate.
10. Show the RELEASE or BLOCK result and the evidence hash/request ID.
11. Open **Attestations** to show the decision trail.

## Safe speaking claims

- “The raw biometric does not need to leave the patient device.”
- “The system decides what representation is necessary for the authorized task.”
- “A representation is tested for residual identity exposure and task utility before release.”
- “If the evidence is insufficient, the system fails closed.”
- “The current prototype demonstrates the release-control architecture; clinical validation is future work.”

## Claims to avoid

- Do not say the system guarantees anonymity.
- Do not say it is clinically validated.
- Do not say it is HIPAA/GDPR compliant solely from this prototype.
- Do not say the current browser demo runs the full attacker ensemble live.
- Do not present simulated or intermediate experiment values as final results.

## Final result policy
Only use values from the final locked benchmark artifact after the model/evaluation protocol has been frozen.
