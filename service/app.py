"""Production-oriented TAPF-MIN release-decision service.

Boundary: this service accepts measured privacy/utility evidence only. It does not
accept, persist, or relay raw biometric frames. Edge/on-device processing remains the
preferred deployment topology.
"""
from datetime import datetime, timezone
import os
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tapf.controller import ReleasePolicy, MinimumDisclosureController
from tapf.deployment import BoundedEvidence, DeploymentPolicy, allowed_representations, select_minimum_release
from tapf.signing import sign_attestation

app = FastAPI(title="TAPF-MIN Release Service", version="0.2.0")


class OperatingPoint(BaseModel):
    alpha: float = Field(ge=0.0, le=1.0)
    identity_risk: float = Field(ge=0.0, le=1.0)
    task_utility: float = Field(ge=0.0, le=1.0)
    temporal_risk: float = Field(ge=0.0, le=1.0)


class ReleaseRequest(BaseModel):
    task: str = Field(min_length=1, max_length=128)
    representation: str = Field(min_length=1, max_length=128)
    measured_at: datetime
    max_evidence_age_seconds: int = Field(default=300, ge=1, le=86400)
    privacy_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    utility_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    temporal_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    operating_points: List[OperatingPoint] = Field(min_length=1)


class BoundedOperatingPoint(BaseModel):
    alpha: float = Field(ge=0.0, le=1.0)
    identity_risk: float = Field(ge=0.0, le=1.0)
    identity_risk_upper: float = Field(ge=0.0, le=1.0)
    task_utility: float = Field(ge=0.0, le=1.0)
    task_utility_lower: float = Field(ge=0.0, le=1.0)
    temporal_risk: float = Field(ge=0.0, le=1.0)
    temporal_risk_upper: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0, le=60000.0)
    evaluator_id: str = Field(min_length=1, max_length=256)
    sample_count: int = Field(ge=1)


class BoundedReleaseRequest(BaseModel):
    task: str = Field(min_length=1, max_length=128)
    representation: str = Field(min_length=1, max_length=128)
    measured_at: datetime
    max_evidence_age_seconds: int = Field(default=300, ge=1, le=86400)
    privacy_upper_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    utility_lower_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    temporal_upper_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    latency_ms_threshold: float = Field(default=150.0, gt=0.0, le=60000.0)
    operating_points: List[BoundedOperatingPoint] = Field(min_length=1)


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _age_or_block(measured_at: datetime, max_age: int):
    measured = _utc(measured_at).astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    age = (now - measured).total_seconds()
    if age < -60:
        raise HTTPException(status_code=422, detail="Evidence timestamp is in the future")
    if age > max_age:
        return age, {
            "decision": "BLOCK",
            "reason": "stale_evidence",
            "evidence_age_seconds": age,
            "selected_alpha": None,
        }
    return age, None


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "service": "tapf-min-release",
        "version": app.version,
        "raw_biometric_ingestion": False,
        "fail_closed": True,
        "bounded_release_api": True,
        "attestation_signing_configured": bool(os.getenv("TAPF_ATTESTATION_SECRET")),
    }


@app.get("/v1/tasks/{task}/representations")
def representations(task: str):
    return {"task": task, "allowed_representations": allowed_representations(task)}


@app.post("/v1/release/evaluate")
def evaluate_release(req: ReleaseRequest):
    """Legacy point-estimate API retained for research compatibility.

    Production deployments should use /v2/release/evaluate with confidence bounds.
    """
    age, blocked = _age_or_block(req.measured_at, req.max_evidence_age_seconds)
    if blocked:
        return blocked

    rows = sorted(req.operating_points, key=lambda x: x.alpha)
    lookup = {round(r.alpha, 8): r for r in rows}
    policy = ReleasePolicy(
        privacy_threshold=req.privacy_threshold,
        utility_threshold=req.utility_threshold,
        temporal_threshold=req.temporal_threshold,
        alphas=tuple(sorted(lookup)),
        fail_closed=True,
    )

    def evaluator(alpha):
        r = lookup[round(float(alpha), 8)]
        return r.identity_risk, r.task_utility, r.temporal_risk

    decision = MinimumDisclosureController(policy).search(evaluator)
    return {
        **decision,
        "task": req.task,
        "representation": req.representation,
        "evidence_age_seconds": age,
        "raw_biometric_ingestion": False,
        "production_recommendation": "Use /v2/release/evaluate with confidence bounds and latency evidence.",
    }


@app.post("/v2/release/evaluate")
def evaluate_bounded_release(req: BoundedReleaseRequest):
    age, blocked = _age_or_block(req.measured_at, req.max_evidence_age_seconds)
    if blocked:
        return blocked

    policy = DeploymentPolicy(
        privacy_upper_threshold=req.privacy_upper_threshold,
        utility_lower_threshold=req.utility_lower_threshold,
        temporal_upper_threshold=req.temporal_upper_threshold,
        latency_ms_threshold=req.latency_ms_threshold,
        fail_closed=True,
    )
    evidence = [BoundedEvidence(**row.model_dump()) for row in req.operating_points]
    decision = select_minimum_release(req.task, req.representation, evidence, policy)

    attestation = {
        "schema": "tapf-min-attestation/v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": req.task,
        "representation": req.representation,
        "release_decision": decision["decision"],
        "reason": decision.get("reason"),
        "selected_alpha": decision.get("selected_alpha"),
        "evaluation": decision.get("evaluation"),
        "thresholds": {
            "privacy_upper": policy.privacy_upper_threshold,
            "utility_lower": policy.utility_lower_threshold,
            "temporal_upper": policy.temporal_upper_threshold,
            "latency_ms": policy.latency_ms_threshold,
        },
        "evidence_age_seconds": age,
        "raw_biometric_transmitted": False,
        "scope": "Empirical release decision under configured evaluators and threat model; not formal anonymity or clinical validation.",
    }
    secret = os.getenv("TAPF_ATTESTATION_SECRET")
    if secret:
        attestation = sign_attestation(attestation, secret, os.getenv("TAPF_ATTESTATION_KEY_ID", "local-hmac"))

    return {
        **decision,
        "task": req.task,
        "representation": req.representation,
        "evidence_age_seconds": age,
        "raw_biometric_ingestion": False,
        "attestation": attestation,
    }
