"""Tamper-evident Privacy-Utility Attestation signing.

HMAC signatures provide integrity/authenticity when the verifier shares the secret.
They are not a substitute for hardware-backed remote attestation or PKI.
"""
from __future__ import annotations
import hashlib
import hmac
import json
from typing import Any


def canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign_attestation(payload: dict[str, Any], secret: str, key_id: str = "local-hmac") -> dict[str, Any]:
    if not secret:
        raise ValueError("Attestation secret must be configured")
    unsigned = dict(payload)
    unsigned.pop("signature", None)
    digest = hmac.new(secret.encode("utf-8"), canonical_json(unsigned), hashlib.sha256).hexdigest()
    return {**unsigned, "signature": {"algorithm": "HMAC-SHA256", "key_id": key_id, "digest": digest}}


def verify_attestation(payload: dict[str, Any], secret: str) -> bool:
    signature = payload.get("signature") or {}
    expected = signature.get("digest")
    if not expected or signature.get("algorithm") != "HMAC-SHA256":
        return False
    unsigned = dict(payload)
    unsigned.pop("signature", None)
    actual = hmac.new(secret.encode("utf-8"), canonical_json(unsigned), hashlib.sha256).hexdigest()
    return hmac.compare_digest(actual, expected)
