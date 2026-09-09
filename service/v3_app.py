"""TAPF-MIN v3 fail-closed release service.

Security properties of this reference service:
- client cannot submit utility/privacy evidence values;
- release evidence is loaded from a frozen server-side registry;
- cumulative DP spend is loaded from a persistent SQLite ledger;
- duplicate release IDs are rejected;
- model/protocol/evidence digests are bound into the attestation;
- raw biometric video is not accepted by this API.

The service is intended for a trusted edge node. `privacy_scope` must be derived from an
authenticated subject/session identity by the deployment layer; this reference API does
not claim that an arbitrary caller-provided scope string authenticates a patient.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import uuid

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from tapf.evidence_registry import EvidenceRegistry
from tapf.formal_privacy import FormalPrivacyBudget, gaussian_release, randomized_response
from tapf.persistent_privacy import DuplicateRelease, SQLiteDPAccountant
from tapf.signing import sign_attestation

app = FastAPI(title="TAPF-MIN v3 Edge Release Service", version="0.6.0")


class V3ReleaseRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=8, max_length=128)
    release_id: str = Field(min_length=8, max_length=256)
    privacy_scope: str = Field(min_length=1, max_length=256)
    evidence_id: str = Field(min_length=1, max_length=256)
    task: str = Field(min_length=1, max_length=128)
    local_posterior: list[float] = Field(min_length=2, max_length=256)


def _require_api_key(authorization: str | None):
    expected = os.getenv("TAPF_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="TAPF_API_KEY must be configured")
    supplied = ""
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API credential")


def _registry() -> EvidenceRegistry:
    path = os.getenv("TAPF_EVIDENCE_REGISTRY")
    if not path:
        raise HTTPException(status_code=503, detail="TAPF_EVIDENCE_REGISTRY must be configured")
    try:
        return EvidenceRegistry.from_json(path)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Evidence registry unavailable: {exc}") from exc


def _normalize_posterior(values):
    p = np.asarray(values, dtype=np.float64)
    if not np.isfinite(p).all() or np.any(p < 0) or p.sum() <= 0:
        raise HTTPException(status_code=422, detail="local_posterior must contain finite non-negative mass")
    return p / p.sum()


def _quantize_simplex(values, bits):
    x = np.asarray(values, dtype=np.float64)
    x = np.maximum(x, 0.0)
    x = x / x.sum() if x.sum() > 0 else np.full_like(x, 1/len(x))
    levels = (1 << int(bits)) - 1
    q = np.round(x * levels) / levels
    return q / q.sum() if q.sum() > 0 else np.full_like(q, 1/len(q))


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "version": app.version,
        "raw_biometric_ingestion": False,
        "server_side_evidence": bool(os.getenv("TAPF_EVIDENCE_REGISTRY")),
        "persistent_privacy_ledger": bool(os.getenv("TAPF_PRIVACY_DB")),
        "auth_configured": bool(os.getenv("TAPF_API_KEY")),
        "attestation_configured": bool(os.getenv("TAPF_ATTESTATION_SECRET")),
    }


@app.post("/v3/release")
def release(req: V3ReleaseRequest, authorization: str | None = Header(default=None)):
    _require_api_key(authorization)
    registry = _registry()
    try:
        evidence = registry.get(req.evidence_id)
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if evidence.task != req.task:
        raise HTTPException(status_code=403, detail="evidence/task mismatch")

    p = _normalize_posterior(req.local_posterior)
    db_path = os.getenv("TAPF_PRIVACY_DB")
    if not db_path:
        raise HTTPException(status_code=503, detail="TAPF_PRIVACY_DB must be configured")
    eps_max = float(os.getenv("TAPF_EPSILON_BUDGET", "4.0"))
    delta_max = float(os.getenv("TAPF_DELTA_BUDGET", "1e-5"))
    ledger = SQLiteDPAccountant(db_path, req.privacy_scope, FormalPrivacyBudget(eps_max, delta_max))

    rng = np.random.default_rng()
    classes = tuple(os.getenv("TAPF_TASK_CLASSES", "ANG,DIS,FEA,HAP,NEU,SAD").split(","))
    if len(classes) != len(p):
        raise HTTPException(status_code=422, detail="posterior/class count mismatch")

    try:
        if evidence.mechanism == "randomized_response_label":
            local_label = classes[int(np.argmax(p))]
            mechanism = randomized_response(local_label, classes, epsilon=evidence.epsilon, rng=rng)
            payload = {"type": "categorical_label", "value": mechanism["label"]}
            ledger.spend_once(req.release_id, mechanism["mechanism"], evidence.epsilon, evidence.delta)
        elif evidence.mechanism == "gaussian_quantized_posterior":
            mechanism = gaussian_release(p, epsilon=evidence.epsilon, delta=evidence.delta,
                                         clip_norm=1.0, rng=rng)
            qbits = max(1, evidence.disclosure_bits // len(classes))
            payload = {"type": "quantized_posterior",
                       "values": _quantize_simplex(mechanism["values"], qbits).tolist(),
                       "quantization_bits": qbits}
            mechanism = {**mechanism, "values": None}
            ledger.spend_once(req.release_id, "gaussian_bounded_task_vector", evidence.epsilon, evidence.delta)
        else:
            raise HTTPException(status_code=403, detail="unsupported approved mechanism")
    except DuplicateRelease as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=403, detail=f"release blocked: {exc}") from exc

    attestation = {
        "schema": "tapf-min-attestation/v5",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "request_id": req.request_id,
        "release_id": req.release_id,
        "privacy_scope_sha256": hashlib.sha256(req.privacy_scope.encode()).hexdigest(),
        "task": req.task,
        "evidence_id": evidence.evidence_id,
        "evidence_sha256": evidence.canonical_digest(),
        "registry_sha256": registry.snapshot_digest(),
        "model_sha256": evidence.model_sha256,
        "protocol_sha256": evidence.protocol_sha256,
        "representation": evidence.representation,
        "mechanism": evidence.mechanism,
        "epsilon": evidence.epsilon,
        "delta": evidence.delta,
        "disclosure_bits": evidence.disclosure_bits,
        "formal_privacy": ledger.statement(),
        "raw_biometric_transmitted": False,
        "latent_transmitted": False,
        "scope": "DP guarantee applies to the released task object under the registered mechanism and accounting scope; not anonymity or clinical validation.",
    }
    secret = os.getenv("TAPF_ATTESTATION_SECRET")
    if not secret:
        raise HTTPException(status_code=503, detail="TAPF_ATTESTATION_SECRET must be configured")
    attestation = sign_attestation(attestation, secret, os.getenv("TAPF_ATTESTATION_KEY_ID", "edge-hmac-v3"))
    return {"decision": "RELEASE", "payload": payload, "mechanism": mechanism, "attestation": attestation}
