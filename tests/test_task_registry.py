from tapf.deployment import allowed_representations, is_known_task, task_registry


def test_task_registry_contains_current_supported_task_families():
    tasks = task_registry()
    expected = {
        'emotion', 'facial-expression', 'movement', 'neurology-motion',
        'rppg', 'authentication', 'clinician-visual'
    }
    assert expected.issubset(tasks)
    for task in expected:
        assert tasks[task]['label']
        assert tasks[task]['purpose_example']
        assert tasks[task]['preferred_representation'] in tasks[task]['representations']


def test_unknown_tasks_fail_closed_with_no_representation_fallback():
    assert is_known_task('unknown-clinical-task') is False
    assert allowed_representations('unknown-clinical-task') == ()


def test_current_minimum_representation_preferences():
    tasks = task_registry()
    assert tasks['emotion']['preferred_representation'] == 'action-units'
    assert tasks['movement']['preferred_representation'] == 'motion-features'
    assert tasks['neurology-motion']['preferred_representation'] == 'motion-features'
    assert tasks['rppg']['preferred_representation'] == 'physiological-signal'
    assert tasks['authentication']['preferred_representation'] == 'cancelable-template'
    assert tasks['clinician-visual']['representations'] == ['protected-video']
