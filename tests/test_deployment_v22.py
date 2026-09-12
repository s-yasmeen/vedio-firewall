from tapf.deployment_v22 import V22BoundedEvidence, V22DeploymentPolicy, select_v22_release


def evidence(**overrides):
    base = dict(
        alpha=0.5,
        clip_auc_ci95_low=0.48,
        clip_auc_ci95_high=0.54,
        repeated_release_auc=0.53,
        task_f1_ci95_low=0.24,
        task_f1_ci95_high=0.31,
        attacker_aucs=(0.51, 0.52, 0.49, 0.53),
        latency_ms=80.0,
        evaluator_id="test-ensemble",
        sample_count=200,
    )
    base.update(overrides)
    return V22BoundedEvidence(**base)


def test_v22_release_when_all_constraints_pass():
    out = select_v22_release("emotion", "expression-embedding", [evidence()])
    assert out["decision"] == "RELEASE"


def test_v22_blocks_repeated_release_leakage():
    out = select_v22_release("emotion", "expression-embedding", [evidence(repeated_release_auc=0.63)])
    assert out["decision"] == "BLOCK"


def test_v22_blocks_reversed_auc_leakage():
    # AUC 0.40 is effectively 0.60 after reversing score orientation.
    out = select_v22_release(
        "emotion", "expression-embedding",
        [evidence(clip_auc_ci95_low=0.39, clip_auc_ci95_high=0.47, attacker_aucs=(0.40, 0.52))],
    )
    assert out["decision"] == "BLOCK"


def test_v22_blocks_weak_utility():
    out = select_v22_release("emotion", "expression-embedding", [evidence(task_f1_ci95_low=0.19)])
    assert out["decision"] == "BLOCK"


def test_v22_blocks_unauthorized_representation():
    out = select_v22_release("authentication", "expression-embedding", [evidence()])
    assert out["decision"] == "BLOCK"


def test_v22_blocks_incomplete_evidence():
    out = select_v22_release("emotion", "expression-embedding", [evidence(sample_count=0)])
    assert out["decision"] == "BLOCK"
