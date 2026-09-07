"""Production-oriented TAPF-MIN release-decision service.

Boundary: this service accepts measured privacy/utility evidence only. It does not
accept, persist, or relay raw biometric frames. Edge/on-device processing remains the
preferred deployment topology.
"""
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import uuid
from typing import List

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tapf.controller import ReleasePolicy, MinimumDisclosureController
from tapf.deployment import BoundedEvidence, DeploymentPolicy, allowed_representations, select_minimum_release
from tapf.signing import sign_attestation

app = FastAPI(title="TAPF-MIN Release Service", version="0.4.0")
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


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
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=8, max_length=128)
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
        return age, {"decision":"BLOCK","reason":"stale_evidence","evidence_age_seconds":age,"selected_alpha":None}
    return age, None


def _require_api_key(authorization: str | None):
    expected = os.getenv("TAPF_API_KEY")
    if not expected:
        return
    supplied = ""
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API credential")


def _evidence_digest(req: BoundedReleaseRequest) -> str:
    payload = req.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/app/")


@app.get("/healthz")
def healthz():
    return {
        "status":"ok","service":"tapf-min-release","version":app.version,
        "raw_biometric_ingestion":False,"fail_closed":True,"bounded_release_api":True,
        "ui_available":FRONTEND_DIR.exists(),
        "attestation_signing_configured":bool(os.getenv("TAPF_ATTESTATION_SECRET")),
        "api_auth_configured":bool(os.getenv("TAPF_API_KEY")),
    }


@app.get("/readyz")
def readyz():
    checks = {
        "attestation_signing": bool(os.getenv("TAPF_ATTESTATION_SECRET")),
        "api_authentication": bool(os.getenv("TAPF_API_KEY")),
    }
    ready = all(checks.values())
    if not ready:
        raise HTTPException(status_code=503, detail={"ready":False,"checks":checks})
    return {"ready":True,"checks":checks,"raw_biometric_ingestion":False}


@app.get("/v1/tasks/{task}/representations")
def representations(task: str):
    return {"task":task,"allowed_representations":allowed_representations(task)}


@app.post("/v1/release/evaluate")
def evaluate_release(req: ReleaseRequest):
    """Legacy point-estimate API retained for research compatibility only."""
    age, blocked = _age_or_block(req.measured_at, req.max_evidence_age_seconds)
    if blocked: return blocked
    rows=sorted(req.operating_points,key=lambda x:x.alpha); lookup={round(r.alpha,8):r for r in rows}
    policy=ReleasePolicy(req.privacy_threshold,req.utility_threshold,req.temporal_threshold,tuple(sorted(lookup)),True)
    def evaluator(alpha):
        r=lookup[round(float(alpha),8)]; return r.identity_risk,r.task_utility,r.temporal_risk
    decision=MinimumDisclosureController(policy).search(evaluator)
    return {**decision,"task":req.task,"representation":req.representation,"evidence_age_seconds":age,
            "raw_biometric_ingestion":False,"production_recommendation":"Use authenticated /v2/release/evaluate with confidence bounds and latency evidence."}


@app.post("/v2/release/evaluate")
def evaluate_bounded_release(req: BoundedReleaseRequest, authorization: str | None = Header(default=None)):
    _require_api_key(authorization)
    age, blocked = _age_or_block(req.measured_at, req.max_evidence_age_seconds)
    if blocked: return blocked

    policy=DeploymentPolicy(req.privacy_upper_threshold,req.utility_lower_threshold,req.temporal_upper_threshold,req.latency_ms_threshold,True)
    evidence=[BoundedEvidence(**row.model_dump()) for row in req.operating_points]
    decision=select_minimum_release(req.task,req.representation,evidence,policy)
    digest=_evidence_digest(req)
    attestation={
        "schema":"tapf-min-attestation/v3","timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "request_id":req.request_id,"evidence_sha256":digest,
        "task":req.task,"representation":req.representation,
        "release_decision":decision["decision"],"reason":decision.get("reason"),
        "selected_alpha":decision.get("selected_alpha"),"evaluation":decision.get("evaluation"),
        "thresholds":{"privacy_upper":policy.privacy_upper_threshold,"utility_lower":policy.utility_lower_threshold,
                      "temporal_upper":policy.temporal_upper_threshold,"latency_ms":policy.latency_ms_threshold},
        "evidence_age_seconds":age,"raw_biometric_transmitted":False,
        "scope":"Empirical release decision under configured evaluators and threat model; not formal anonymity or clinical validation.",
    }
    secret=os.getenv("TAPF_ATTESTATION_SECRET")
    if secret: attestation=sign_attestation(attestation,secret,os.getenv("TAPF_ATTESTATION_KEY_ID","local-hmac"))
    return {**decision,"request_id":req.request_id,"evidence_sha256":digest,"task":req.task,"representation":req.representation,
            "evidence_age_seconds":age,"raw_biometric_ingestion":False,"attestation":attestation}


if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="tapf-min-ui")
