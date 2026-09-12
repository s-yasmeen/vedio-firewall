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
from tapf.deployment_v22 import (
    V22BoundedEvidence,
    V22DeploymentPolicy,
    select_authorized_v22_release,
    select_v22_release,
)
from tapf.task_contract import TaskAuthorizationContract
from tapf.signing import sign_attestation

app = FastAPI(title="TAPF-MIN Release Service", version="0.7.0")
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
    privacy_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    utility_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    temporal_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
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
    privacy_upper_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    utility_lower_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    temporal_upper_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    latency_ms_threshold: float = Field(default=150.0, gt=0.0, le=60000.0)
    operating_points: List[BoundedOperatingPoint] = Field(min_length=1)
    prior_release_count: int = Field(default=0, ge=0, le=1000000)
    composition_risk_upper: float | None = Field(default=None, ge=0.0, le=1.0)
    composition_upper_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    composition_evaluator_id: str | None = Field(default=None, max_length=256)


class V22OperatingPoint(BaseModel):
    alpha: float = Field(ge=0.0, le=1.0)
    clip_auc_ci95_low: float = Field(ge=0.0, le=1.0)
    clip_auc_ci95_high: float = Field(ge=0.0, le=1.0)
    repeated_release_auc: float = Field(ge=0.0, le=1.0)
    task_f1_ci95_low: float = Field(ge=0.0, le=1.0)
    task_f1_ci95_high: float = Field(ge=0.0, le=1.0)
    attacker_aucs: List[float] = Field(min_length=1)
    latency_ms: float = Field(ge=0.0, le=60000.0)
    evaluator_id: str = Field(min_length=1, max_length=256)
    sample_count: int = Field(ge=1)


class V22ReleaseRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=8, max_length=128)
    task: str = Field(min_length=1, max_length=128)
    representation: str = Field(min_length=1, max_length=128)
    measured_at: datetime
    max_evidence_age_seconds: int = Field(default=300, ge=1, le=86400)
    max_identity_advantage: float = Field(default=0.05, ge=0.0, le=0.50)
    min_task_f1_lower_ci: float = Field(default=0.20, ge=0.0, le=1.0)
    latency_ms_threshold: float = Field(default=150.0, gt=0.0, le=60000.0)
    operating_points: List[V22OperatingPoint] = Field(min_length=1)


class TaskContractInput(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    task: str = Field(min_length=1, max_length=128)
    purpose: str = Field(min_length=1, max_length=512)
    recipient_id: str = Field(min_length=1, max_length=256)
    requested_representation: str = Field(min_length=1, max_length=128)
    patient_authorized: bool
    issued_at: datetime
    expires_at: datetime | None = None

    def to_domain(self) -> TaskAuthorizationContract:
        return TaskAuthorizationContract(
            session_id=self.session_id,
            task=self.task,
            purpose=self.purpose,
            recipient_id=self.recipient_id,
            requested_representation=self.requested_representation,
            patient_authorized=self.patient_authorized,
            issued_at=self.issued_at.isoformat(),
            expires_at=self.expires_at.isoformat() if self.expires_at else None,
        )


class AuthorizedV22ReleaseRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=8, max_length=128)
    contract: TaskContractInput
    measured_at: datetime
    max_evidence_age_seconds: int = Field(default=300, ge=1, le=86400)
    max_identity_advantage: float = Field(default=0.05, ge=0.0, le=0.50)
    min_task_f1_lower_ci: float = Field(default=0.20, ge=0.0, le=1.0)
    latency_ms_threshold: float = Field(default=150.0, gt=0.0, le=60000.0)
    operating_points: List[V22OperatingPoint] = Field(min_length=1)


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


def _digest_payload(payload) -> str:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evidence_digest(req: BoundedReleaseRequest) -> str:
    return _digest_payload(req)


def _composition_block(req: BoundedReleaseRequest):
    """Fail closed on repeated disclosure without bounded composition evidence."""
    if req.prior_release_count == 0:
        return None
    if req.composition_risk_upper is None or not req.composition_evaluator_id:
        return {
            "decision":"BLOCK","reason":"composition_evidence_required",
            "selected_alpha":None,"prior_release_count":req.prior_release_count,
            "composition_risk_upper":req.composition_risk_upper,
        }
    if req.composition_risk_upper > req.composition_upper_threshold:
        return {
            "decision":"BLOCK","reason":"composition_risk_exceeds_threshold",
            "selected_alpha":None,"prior_release_count":req.prior_release_count,
            "composition_risk_upper":req.composition_risk_upper,
            "composition_upper_threshold":req.composition_upper_threshold,
        }
    return None


def _v22_evidence(rows: List[V22OperatingPoint]) -> list[V22BoundedEvidence]:
    return [V22BoundedEvidence(
        alpha=row.alpha,
        clip_auc_ci95_low=row.clip_auc_ci95_low,
        clip_auc_ci95_high=row.clip_auc_ci95_high,
        repeated_release_auc=row.repeated_release_auc,
        task_f1_ci95_low=row.task_f1_ci95_low,
        task_f1_ci95_high=row.task_f1_ci95_high,
        attacker_aucs=tuple(row.attacker_aucs),
        latency_ms=row.latency_ms,
        evaluator_id=row.evaluator_id,
        sample_count=row.sample_count,
    ) for row in rows]


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/app/")


@app.get("/healthz")
def healthz():
    return {
        "status":"ok","service":"tapf-min-release","version":app.version,
        "raw_biometric_ingestion":False,"fail_closed":True,"bounded_release_api":True,
        "v22_chance_centered_gate":True,"composition_guard":True,
        "two_sided_task_authorization":True,
        "preferred_release_endpoint":"/v23/authorized-release/evaluate",
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
            "raw_biometric_ingestion":False,"production_recommendation":"Prefer authenticated /v23/authorized-release/evaluate."}


@app.post("/v2/release/evaluate")
def evaluate_bounded_release(req: BoundedReleaseRequest, authorization: str | None = Header(default=None)):
    """Compatibility endpoint using normalized empirical risk/utility bounds."""
    _require_api_key(authorization)
    age, blocked = _age_or_block(req.measured_at, req.max_evidence_age_seconds)
    if blocked: return blocked
    composition_block = _composition_block(req)
    if composition_block:
        return {**composition_block,"request_id":req.request_id,"task":req.task,"representation":req.representation,
                "evidence_age_seconds":age,"raw_biometric_ingestion":False,"composition_guard":True}

    policy=DeploymentPolicy(req.privacy_upper_threshold,req.utility_lower_threshold,req.temporal_upper_threshold,req.latency_ms_threshold,True)
    evidence=[BoundedEvidence(**row.model_dump()) for row in req.operating_points]
    decision=select_minimum_release(req.task,req.representation,evidence,policy)
    digest=_evidence_digest(req)
    attestation={
        "schema":"tapf-min-attestation/v4","timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "request_id":req.request_id,"evidence_sha256":digest,
        "task":req.task,"representation":req.representation,
        "release_decision":decision["decision"],"reason":decision.get("reason"),
        "selected_alpha":decision.get("selected_alpha"),"evaluation":decision.get("evaluation"),
        "thresholds":{"privacy_upper":policy.privacy_upper_threshold,"utility_lower":policy.utility_lower_threshold,
                      "temporal_upper":policy.temporal_upper_threshold,"latency_ms":policy.latency_ms_threshold,
                      "composition_upper":req.composition_upper_threshold},
        "composition":{"prior_release_count":req.prior_release_count,
                       "risk_upper":req.composition_risk_upper,
                       "evaluator_id":req.composition_evaluator_id,
                       "guard_pass":True},
        "evidence_age_seconds":age,"raw_biometric_transmitted":False,
        "scope":"Empirical normalized-risk release decision; not formal anonymity, differential privacy, or clinical validation.",
    }
    secret=os.getenv("TAPF_ATTESTATION_SECRET")
    if secret: attestation=sign_attestation(attestation,secret,os.getenv("TAPF_ATTESTATION_KEY_ID","local-hmac"))
    return {**decision,"request_id":req.request_id,"evidence_sha256":digest,"task":req.task,"representation":req.representation,
            "evidence_age_seconds":age,"raw_biometric_ingestion":False,"composition_guard":True,"attestation":attestation}


@app.post("/v22/release/evaluate")
def evaluate_v22_release(req: V22ReleaseRequest, authorization: str | None = Header(default=None)):
    """Compatibility v2.2 endpoint using direct chance-centered AUC and F1 evidence."""
    _require_api_key(authorization)
    age, blocked = _age_or_block(req.measured_at, req.max_evidence_age_seconds)
    if blocked: return blocked

    policy = V22DeploymentPolicy(
        max_identity_advantage=req.max_identity_advantage,
        min_task_f1_lower_ci=req.min_task_f1_lower_ci,
        latency_ms_threshold=req.latency_ms_threshold,
        fail_closed=True,
    )
    evidence = _v22_evidence(req.operating_points)
    decision = select_v22_release(req.task, req.representation, evidence, policy)
    digest = _digest_payload(req)
    attestation = {
        "schema":"tapf-min-attestation/v5-v22",
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "request_id":req.request_id,
        "evidence_sha256":digest,
        "task":req.task,
        "representation":req.representation,
        "release_decision":decision["decision"],
        "reason":decision.get("reason"),
        "selected_alpha":decision.get("selected_alpha"),
        "evaluation":decision.get("evaluation"),
        "thresholds":{
            "chance_identity_auc":0.5,
            "max_identity_advantage":policy.max_identity_advantage,
            "max_effective_auc":policy.max_effective_auc,
            "min_task_f1_lower_ci":policy.min_task_f1_lower_ci,
            "latency_ms":policy.latency_ms_threshold,
        },
        "evidence_age_seconds":age,
        "raw_biometric_transmitted":False,
        "scope":"Empirical v2.2 release decision under tested threat model; not formal anonymity, differential privacy, clinical validation, diagnostic efficacy, or regulatory compliance.",
    }
    secret=os.getenv("TAPF_ATTESTATION_SECRET")
    if secret: attestation=sign_attestation(attestation,secret,os.getenv("TAPF_ATTESTATION_KEY_ID","local-hmac"))
    return {**decision,"request_id":req.request_id,"evidence_sha256":digest,
            "task":req.task,"representation":req.representation,"evidence_age_seconds":age,
            "raw_biometric_ingestion":False,"attestation":attestation}


@app.post("/v23/authorized-release/evaluate")
def evaluate_authorized_v23_release(
    req: AuthorizedV22ReleaseRequest,
    authorization: str | None = Header(default=None),
    recipient_id: str = Header(alias="X-TAPF-Recipient-ID"),
):
    """Two-sided doctor-request/patient-authorization release gate.

    The clinical side declares task, purpose, recipient and requested representation.
    The patient side must authorize that exact contract. Only then is measured v2.2
    privacy/utility/repeated-release evidence evaluated. Recipient binding is supplied
    separately from the payload so a request cannot self-validate by changing both
    payload and expected-recipient fields together. In production the recipient binding
    should be derived from the authenticated principal.
    """
    _require_api_key(authorization)
    age, blocked = _age_or_block(req.measured_at, req.max_evidence_age_seconds)
    if blocked:
        return {**blocked, "request_id": req.request_id, "task_authorization_pass": False}

    policy = V22DeploymentPolicy(
        max_identity_advantage=req.max_identity_advantage,
        min_task_f1_lower_ci=req.min_task_f1_lower_ci,
        latency_ms_threshold=req.latency_ms_threshold,
        fail_closed=True,
    )
    contract = req.contract.to_domain()
    evidence = _v22_evidence(req.operating_points)
    decision = select_authorized_v22_release(
        contract,
        evidence,
        expected_recipient_id=recipient_id,
        policy=policy,
    )
    digest = _digest_payload(req)
    contract_result = decision.get("contract", {})
    task = contract_result.get("task", req.contract.task.strip().lower())
    representation = contract_result.get(
        "requested_representation", req.contract.requested_representation.strip().lower()
    )
    attestation = {
        "schema":"tapf-min-attestation/v6-task-aware",
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "request_id":req.request_id,
        "evidence_sha256":digest,
        "contract_sha256":contract_result.get("contract_digest", contract.digest()),
        "session_id":req.contract.session_id,
        "task":task,
        "purpose":req.contract.purpose,
        "recipient_id":req.contract.recipient_id,
        "recipient_binding":recipient_id,
        "patient_authorized":req.contract.patient_authorized,
        "representation":representation,
        "task_authorization_pass":bool(decision.get("task_authorization_pass", False)),
        "authorization_checks":contract_result.get("checks", {}),
        "release_decision":decision["decision"],
        "reason":decision.get("reason"),
        "selected_alpha":decision.get("selected_alpha"),
        "evaluation":decision.get("evaluation"),
        "thresholds":{
            "chance_identity_auc":0.5,
            "max_identity_advantage":policy.max_identity_advantage,
            "max_effective_auc":policy.max_effective_auc,
            "min_task_f1_lower_ci":policy.min_task_f1_lower_ci,
            "latency_ms":policy.latency_ms_threshold,
        },
        "evidence_age_seconds":age,
        "raw_biometric_transmitted":False,
        "scope":"Task-conditioned empirical release decision under tested threat model. Recipient header is a demo binding; production deployments should bind recipient identity to authenticated credentials. Not formal anonymity, differential privacy, clinical validation, diagnostic efficacy, consent management, or regulatory compliance.",
    }
    secret=os.getenv("TAPF_ATTESTATION_SECRET")
    if secret:
        attestation=sign_attestation(attestation,secret,os.getenv("TAPF_ATTESTATION_KEY_ID","local-hmac"))
    return {
        **decision,
        "request_id":req.request_id,
        "evidence_sha256":digest,
        "contract_sha256":attestation["contract_sha256"],
        "task":task,
        "purpose":req.contract.purpose,
        "recipient_id":req.contract.recipient_id,
        "representation":representation,
        "evidence_age_seconds":age,
        "raw_biometric_ingestion":False,
        "two_sided_task_authorization":True,
        "attestation":attestation,
    }


if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="tapf-min-ui")
