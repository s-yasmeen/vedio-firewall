from tapf.prevalidation_gate import evaluate_prevalidation


def good():
    return {
        'macro_f1': .62, 'macro_f1_lcb': .56,
        'face_valid_rate': .96, 'face_detection_rate': .91,
        'eye_alignment_rate': .58, 'bbox_reuse_rate': .08,
        'actor_detection_gap': .08, 'audit_actor_count': 16,
        'identity_auc_ucb': .57, 'repeated_auc_ucb': .58,
        'verification_auc_ucb': .56, 'linkage_auc_ucb': .58,
        'actor_overlap': 0, 'single_frozen_model': True,
        'selection_separate_from_final': True,
        'actor_clustered_utility_ci': True,
        'identity_level_privacy_ci': True,
        'formal_dp_applied': True, 'raw_video_released': False,
        'latent_released': False,
    }


def test_complete_good_evidence_passes():
    assert evaluate_prevalidation(good())['decision'] == 'PASS'


def test_missing_evidence_blocks():
    e = good(); e.pop('linkage_auc_ucb')
    assert evaluate_prevalidation(e)['decision'] == 'BLOCK'


def test_utility_without_privacy_blocks():
    e = good(); e['macro_f1'] = .90; e['identity_auc_ucb'] = .80
    assert evaluate_prevalidation(e)['decision'] == 'BLOCK'


def test_preprocessing_disparity_or_reuse_blocks():
    e = good(); e['actor_detection_gap'] = .30
    assert evaluate_prevalidation(e)['decision'] == 'BLOCK'
    e = good(); e['bbox_reuse_rate'] = .35
    assert evaluate_prevalidation(e)['decision'] == 'BLOCK'


def test_small_audit_cohort_blocks():
    e = good(); e['audit_actor_count'] = 8
    assert evaluate_prevalidation(e)['decision'] == 'BLOCK'


def test_selection_leak_blocks():
    e = good(); e['selection_separate_from_final'] = False
    assert evaluate_prevalidation(e)['decision'] == 'BLOCK'


def test_nonhierarchical_privacy_ci_blocks():
    e = good(); e['identity_level_privacy_ci'] = False
    assert evaluate_prevalidation(e)['decision'] == 'BLOCK'


def test_raw_or_latent_release_blocks():
    e = good(); e['latent_released'] = True
    assert evaluate_prevalidation(e)['decision'] == 'BLOCK'
