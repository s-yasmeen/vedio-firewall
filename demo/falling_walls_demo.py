"""TAPF-MIN Falling Walls Lab executable demo.

This app demonstrates the *release architecture* with the repository's actual privacy
mechanisms and frozen reported measurements. It deliberately does not fabricate an FER
checkpoint. Until a trained checkpoint is supplied, the six-class posterior shown in the
UI is an explicitly labelled local/demo posterior used only to exercise the privacy layer.

Run:
    streamlit run demo/falling_walls_demo.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tapf.formal_privacy import randomized_response

EMOTIONS = ("ANG", "DIS", "FEA", "HAP", "NEU", "SAD")
EMOTION_NAMES = {
    "ANG": "Anger",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happiness",
    "NEU": "Neutral",
    "SAD": "Sadness",
}

# Frozen, already-measured research evidence. Do not edit these numbers to improve optics.
FROZEN_RESULTS = {
    "TAPF-MIN v2.1 frozen validation": {
        "clips": 144,
        "actors": 24,
        "emotion_macro_f1": 0.230,
        "emotion_f1_ci95": "0.165–0.297",
        "clip_identity_auc": 0.573,
        "identity_auc_ci95": "0.474–0.665",
        "repeated_release_identity_auc": 0.693,
        "release_decision": "BLOCK",
        "note": "Frozen baseline. No eligible operating point.",
    },
    "v3 deep-FER development run": {
        "clips": 144,
        "actors": 24,
        "emotion_accuracy": 0.1597,
        "emotion_macro_f1": 0.1505,
        "emotion_f1_ci95": "0.091–0.212",
        "posterior_identity_auc_warning": "≈0.998 (confounded OOF audit; not a clean leakage estimate)",
        "release_decision": "BLOCK",
        "note": "Development diagnostic only. Actor-disjoint FER remained near chance.",
    },
    "v3.1 hardening status": {
        "engineering_ci": "PASS",
        "architecture": "Face-focused RGB + motion residuals + temporal GRU/attention + delayed GRL",
        "privacy_release": "Minimum disclosure + formal DP + persistent composition",
        "performance_status": "Benchmark not yet claimed",
        "release_decision": "BLOCK until empirical gate passes",
        "note": "No fabricated performance result. Final frozen-model audit still required.",
    },
}


def _normalise(values: list[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = np.maximum(x, 0.0)
    return x / x.sum() if x.sum() > 0 else np.full(len(x), 1.0 / len(x))


def _rr_expected_accuracy(epsilon: float, k: int) -> float:
    return math.exp(epsilon) / (math.exp(epsilon) + k - 1)


def _init_state() -> None:
    st.session_state.setdefault("epsilon_spent", 0.0)
    st.session_state.setdefault("release_count", 0)
    st.session_state.setdefault("last_payload", None)


def _release_demo(posterior: np.ndarray, epsilon: float, epsilon_budget: float) -> dict:
    remaining = epsilon_budget - float(st.session_state.epsilon_spent)
    if epsilon <= 0 or epsilon > remaining + 1e-12:
        return {
            "decision": "BLOCK",
            "reason": "privacy_budget_exhausted",
            "remaining_epsilon": max(0.0, remaining),
        }
    local_label = EMOTIONS[int(np.argmax(posterior))]
    rr = randomized_response(local_label, EMOTIONS, epsilon=epsilon, rng=np.random.default_rng(), accountant=None)
    st.session_state.epsilon_spent += float(epsilon)
    st.session_state.release_count += 1
    payload = {
        "type": "categorical_label",
        "value": rr["label"],
        "payload_bits": int(math.ceil(math.log2(len(EMOTIONS)))),
        "raw_video_released": False,
        "task_latent_released": False,
    }
    st.session_state.last_payload = payload
    return {
        "decision": "RELEASE",
        "local_label": local_label,
        "payload": payload,
        "epsilon_spent": st.session_state.epsilon_spent,
        "remaining_epsilon": max(0.0, epsilon_budget - st.session_state.epsilon_spent),
    }


def main() -> None:
    st.set_page_config(page_title="TAPF-MIN | Falling Walls Lab", page_icon="🧱", layout="wide")
    _init_state()

    st.title("🧱 Breaking the Wall of Biometric Exposure")
    st.subheader("TAPF-MIN — Minimum-Disclosure Biometric Privacy Firewall")
    st.caption(
        "Private sensing → local inference → minimum sufficient information → formally bounded release. "
        "This presentation build demonstrates the real release/privacy logic; it does not fabricate FER accuracy."
    )

    live, evidence, architecture = st.tabs(["⚡ Live Firewall", "📊 Measured Evidence", "🧠 Architecture"])

    with live:
        st.info(
            "Stage-safe mode: the posterior below represents a local FER output. A trained FER checkpoint is not "
            "committed to this branch yet, so these values are demonstration inputs—not benchmark predictions."
        )
        left, right = st.columns([1.25, 1])
        with left:
            st.markdown("#### 1. Private local task posterior")
            defaults = [0.06, 0.05, 0.04, 0.61, 0.17, 0.07]
            vals = []
            cols = st.columns(3)
            for idx, code in enumerate(EMOTIONS):
                with cols[idx % 3]:
                    vals.append(
                        st.slider(
                            f"{EMOTION_NAMES[code]} ({code})",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(defaults[idx]),
                            step=0.01,
                            key=f"p_{code}",
                        )
                    )
            p = _normalise(vals)
            local_idx = int(np.argmax(p))
            st.metric("Local task decision", EMOTION_NAMES[EMOTIONS[local_idx]], f"confidence {p[local_idx]:.1%}")
            st.bar_chart({EMOTION_NAMES[c]: float(v) for c, v in zip(EMOTIONS, p)})

        with right:
            st.markdown("#### 2. Minimum-disclosure release")
            epsilon_budget = st.number_input("Session ε budget", 0.25, 20.0, 4.0, 0.25)
            epsilon = st.select_slider("ε per categorical release", options=[0.25, 0.5, 1.0, 2.0, 4.0], value=1.0)
            expected = _rr_expected_accuracy(float(epsilon), len(EMOTIONS))
            st.caption(
                f"Exact k-ary randomized response. For 6 classes at ε={epsilon:g}, the protected label equals the "
                f"local label with probability {expected:.1%}."
            )
            a, b, c = st.columns(3)
            a.metric("ε spent", f"{st.session_state.epsilon_spent:.2f}")
            b.metric("Releases", int(st.session_state.release_count))
            c.metric("Task payload", "3 bits")

            if st.button("🔐 Request protected release", type="primary", use_container_width=True):
                result = _release_demo(p, float(epsilon), float(epsilon_budget))
                if result["decision"] == "RELEASE":
                    payload = result["payload"]
                    st.success(f"RELEASE → {EMOTION_NAMES[payload['value']]} ({payload['value']})")
                    st.json(payload)
                else:
                    st.error("BLOCK → privacy budget exhausted")
                    st.json(result)

            if st.button("Reset presentation session", use_container_width=True):
                st.session_state.epsilon_spent = 0.0
                st.session_state.release_count = 0
                st.session_state.last_payload = None
                st.rerun()

            st.markdown("#### What crossed the boundary?")
            st.write("✅ Protected task label")
            st.write("❌ Raw video")
            st.write("❌ Face crop")
            st.write("❌ Task embedding / latent")
            st.write("❌ Full biometric template")

    with evidence:
        st.markdown("### Frozen and development evidence")
        st.warning(
            "Scientific rule for the presentation: BLOCK means BLOCK. The app never converts a failed or incomplete "
            "experiment into a privacy claim."
        )
        for title, metrics in FROZEN_RESULTS.items():
            with st.expander(title, expanded=True):
                for key, value in metrics.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")

        st.markdown("### What we can claim today")
        st.markdown(
            "- The architecture prevents raw video and task latents from being authorized release objects.\n"
            "- Categorical release uses exact ε-local differential privacy through k-ary randomized response.\n"
            "- Repeated releases consume a cumulative privacy budget and can fail closed.\n"
            "- The frozen CREMA-D baseline did **not** pass the release gate; this is reported transparently.\n"
            "- The v3.1 engineering hardening CI passes, but improved FER performance still requires a measured benchmark."
        )

    with architecture:
        st.markdown("### System boundary")
        st.code(
            "RAW VIDEO (private device)\n"
            "      │\n"
            "      ▼\n"
            "Face detection + bounded tracking + alignment\n"
            "      │\n"
            "      ▼\n"
            "RGB encoder ─┐\n"
            "             ├─ Gated fusion → GRU → attention → FER posterior\n"
            "Motion CNN ──┘\n"
            "      │\n"
            "      ▼\n"
            "Minimum-disclosure selector\n"
            "      │\n"
            "      ▼\n"
            "Formal DP + persistent composition ledger\n"
            "      │\n"
            "      ├── RELEASE: protected task output only\n"
            "      └── BLOCK: no eligible safe release"
        )
        st.markdown("### Presentation sentence")
        st.success(
            "Instead of sending a patient's face to an AI system, TAPF-MIN keeps the biometric video local and "
            "releases only the minimum task information under a measurable privacy budget."
        )

    st.divider()
    st.caption(
        "Research demonstration — controlled CREMA-D evidence; not clinical validation, not an anonymity guarantee, "
        "and not a claim that differential privacy alone prevents every biometric attack."
    )


if __name__ == "__main__":
    main()
