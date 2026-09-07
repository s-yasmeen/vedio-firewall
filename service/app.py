"""Production-oriented TAPF-MIN release-decision service.

Important boundary: this API accepts measured privacy/utility evidence only. It does NOT
accept or store raw biometric frames. Raw video processing and model inference should
remain on-device/edge-side wherever possible.
"""
from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tapf.controller import ReleasePolicy, MinimumDisclosureController

app = FastAPI(title="TAPF-MIN Release Service", version="0.1.0")


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


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "service": "tapf-min-release",
        "raw_biometric_ingestion": False,
        "fail_closed": True,
    }


@app.post("/v1/release/evaluate")
def evaluate_release(req: ReleaseRequest):
    measured_at = _utc(req.measured_at).astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    age = (now - measured_at).total_seconds()
    if age < -60:
        raise HTTPException(status_code=422, detail="Evidence timestamp is in the future")
    if age > req.max_evidence_age_seconds:
        return {
            "decision": "BLOCK",
            "reason": "stale_evidence",
            "evidence_age_seconds": age,
            "selected_alpha": None,
        }

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
    }
