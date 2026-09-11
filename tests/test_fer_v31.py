import torch
from tapf.fer_v31 import FERV31Config, TAPFMinFERV31, fer_v31_loss


def test_v31_shapes_and_motion_gate():
    cfg=FERV31Config(n_emotions=6,n_identities=8,pretrained=False,temporal_hidden=64,motion_dim=32)
    model=TAPFMinFERV31(cfg)
    x=torch.randn(2,4,3,64,64)
    out=model(x,identity_grl_strength=0.2)
    assert out['emotion_logits'].shape==(2,6)
    assert out['identity_logits'].shape==(2,8)
    assert out['fusion_gate'].shape==(2,4,64)
    assert torch.all(out['fusion_gate']>=0) and torch.all(out['fusion_gate']<=1)
    assert torch.allclose(out['temporal_attention'].sum(1),torch.ones(2),atol=1e-5)


def test_motion_encoder_sequence_shape():
    cfg=FERV31Config(n_identities=4,pretrained=False,temporal_hidden=32,motion_dim=16)
    model=TAPFMinFERV31(cfg)
    x=torch.randn(1,5,3,64,64)
    z=model.encode_motion(x)
    assert z.shape==(1,5,16)


def test_delayed_grl_schedule():
    cfg=FERV31Config(grl_start_fraction=.35,grl_max_strength=.6)
    assert TAPFMinFERV31.scheduled_grl(0.0,cfg)==0.0
    assert TAPFMinFERV31.scheduled_grl(.35,cfg)==0.0
    mid=TAPFMinFERV31.scheduled_grl(.6,cfg)
    late=TAPFMinFERV31.scheduled_grl(1.0,cfg)
    assert 0 < mid < late <= .6 + 1e-9


def test_v31_loss_is_finite():
    cfg=FERV31Config(n_emotions=6,n_identities=4,pretrained=False,temporal_hidden=32,motion_dim=16)
    model=TAPFMinFERV31(cfg)
    x=torch.randn(2,3,3,64,64)
    out=model(x,identity_grl_strength=.1)
    loss=fer_v31_loss(out,torch.tensor([0,1]),torch.tensor([0,1]),identity_weight=.1)
    assert torch.isfinite(loss['loss'])


def test_no_release_latent_api():
    cfg=FERV31Config(n_identities=4,pretrained=False)
    model=TAPFMinFERV31(cfg)
    assert not hasattr(model,'release_latent')
