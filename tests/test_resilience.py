import pytest
from tapf.resilience import (
    SourceHealth, SourceVector, ResiliencePolicy,
    assess_resilience, reliability_weighted_fusion, gate_with_resilience,
)


def test_normal_when_all_sources_healthy():
    out=assess_resilience([
        SourceHealth('video',0.95), SourceHealth('motion',0.90), SourceHealth('rppg',0.86)
    ])
    assert out['mode']=='NORMAL'
    assert out['confidence']>=0.80


def test_degraded_when_one_source_fails_but_remaining_are_reliable():
    out=assess_resilience([
        SourceHealth('video',0.93), SourceHealth('motion',0.88), SourceHealth('rppg',0.10)
    ])
    assert out['mode']=='DEGRADED'
    assert 'rppg' in out['failed_sources']


def test_block_when_required_source_fails():
    out=assess_resilience([
        SourceHealth('video',0.20,required=True), SourceHealth('motion',0.95)
    ])
    assert out['mode']=='BLOCK'
    assert out['reason']=='required_source_unavailable'


def test_block_when_all_sources_unreliable():
    out=assess_resilience([SourceHealth('video',0.20),SourceHealth('motion',0.30)])
    assert out['mode']=='BLOCK'


def test_weighted_fusion_downweights_weaker_source():
    out=reliability_weighted_fusion([
        SourceVector(SourceHealth('motion',0.9),[1.0,0.0]),
        SourceVector(SourceHealth('rppg',0.6),[0.0,1.0]),
    ], ResiliencePolicy(normal_confidence=0.95,degraded_confidence=0.50))
    assert out['mode']=='DEGRADED'
    assert out['effective_weights']['motion']>out['effective_weights']['rppg']
    assert out['fused'][0]>out['fused'][1]


def test_fusion_rejects_dimension_mismatch():
    with pytest.raises(ValueError):
        reliability_weighted_fusion([
            SourceVector(SourceHealth('a',0.9),[1.0,0.0]),
            SourceVector(SourceHealth('b',0.9),[1.0,0.0,0.0]),
        ])


def test_resilience_never_overrides_privacy_block():
    res=assess_resilience([SourceHealth('video',0.95)])
    out=gate_with_resilience(res,{'decision':'BLOCK'})
    assert out['decision']=='BLOCK'
    assert out['reason']=='privacy_or_utility_block'


def test_degraded_can_release_only_after_privacy_utility_release():
    res=assess_resilience([SourceHealth('video',0.90),SourceHealth('rppg',0.1)])
    out=gate_with_resilience(res,{'decision':'RELEASE'})
    assert out['decision']=='RELEASE'
    assert out['operating_mode']=='DEGRADED'
