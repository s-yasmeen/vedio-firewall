import json
import hashlib
import pytest

from tapf.evidence_registry import EvidenceRegistry, FrozenEvidence


def _sha(x: str) -> str:
    return hashlib.sha256(x.encode()).hexdigest()


def _entry(**kw):
    base = dict(
        evidence_id="ev1",
        task="emotion",
        representation="private-label",
        mechanism="randomized_response_label",
        model_sha256=_sha("model"),
        protocol_sha256=_sha("protocol"),
        utility_lower=.80,
        identity_risk_upper=.08,
        repeated_release_risk_upper=.10,
        epsilon=.5,
        delta=0.0,
        disclosure_bits=3,
        status="approved",
    )
    base.update(kw)
    return FrozenEvidence(**base)


def test_registry_selects_only_jointly_eligible_evidence():
    good = _entry()
    bad = _entry(evidence_id="ev2", representation="posterior", disclosure_bits=24,
                 identity_risk_upper=.40)
    r = EvidenceRegistry([good, bad])
    rows = r.eligible(
        task="emotion",
        model_sha256=good.model_sha256,
        protocol_sha256=good.protocol_sha256,
        min_utility_lower=.75,
        max_identity_risk_upper=.15,
        max_repeated_release_risk_upper=.15,
    )
    assert [x.evidence_id for x in rows] == ["ev1"]


def test_registry_rejects_wrong_model_or_protocol():
    good = _entry()
    r = EvidenceRegistry([good])
    assert r.eligible(task="emotion", model_sha256=_sha("other"), protocol_sha256=good.protocol_sha256,
                      min_utility_lower=0, max_identity_risk_upper=1, max_repeated_release_risk_upper=1) == []


def test_nonapproved_evidence_fails_closed():
    r = EvidenceRegistry([_entry(status="blocked")])
    with pytest.raises(PermissionError):
        r.get("ev1")


def test_json_schema_is_checked(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"schema":"wrong","entries":[]}))
    with pytest.raises(ValueError):
        EvidenceRegistry.from_json(p)


def test_snapshot_digest_is_deterministic():
    a = _entry(evidence_id="a")
    b = _entry(evidence_id="b")
    assert EvidenceRegistry([a,b]).snapshot_digest() == EvidenceRegistry([b,a]).snapshot_digest()
