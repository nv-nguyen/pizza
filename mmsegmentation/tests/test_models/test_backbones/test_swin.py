# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.backbones import SwinTransformer


def test_swin_transformer():
    """Test Swin Transformer backbone."""

    with pytest.raises(TypeError):
        # Pretrained arg must be str or None.
        model = SwinTransformer(pretrained=123)

    with pytest.raises(AssertionError):
        # Because swin use non-overlapping patch embed, so the stride of patch
        # embed must be equal to patch size.
        model = SwinTransformer(strides=(2, 2, 2, 2), patch_size=4)

    # Test absolute position embedding
    temp = torch.randn((1, 3, 224, 224))
    model = SwinTransformer(pretrain_img_size=224, use_abs_pos_embed=True)
    model.init_weights()
    model(temp)

    # Test patch norm
    model = SwinTransformer(patch_norm=False)
    model(temp)

    # Test pretrain img size
    model = SwinTransformer(pretrain_img_size=(224, ))

    with pytest.raises(AssertionError):
        model = SwinTransformer(pretrain_img_size=(224, 224, 224))

    # Test normal inference
    temp = torch.randn((1, 3, 512, 512))
    model = SwinTransformer()
    outs = model(temp)
    assert outs[0].shape == (1, 96, 128, 128)
    assert outs[1].shape == (1, 192, 64, 64)
    assert outs[2].shape == (1, 384, 32, 32)
    assert outs[3].shape == (1, 768, 16, 16)

    # Test abnormal inference
    temp = torch.randn((1, 3, 511, 511))
    model = SwinTransformer()
    outs = model(temp)
    assert outs[0].shape == (1, 96, 128, 128)
    assert outs[1].shape == (1, 192, 64, 64)
    assert outs[2].shape == (1, 384, 32, 32)
    assert outs[3].shape == (1, 768, 16, 16)

    # Test abnormal inference
    temp = torch.randn((1, 3, 112, 137))
    model = SwinTransformer()
    outs = model(temp)
    assert outs[0].shape == (1, 96, 28, 35)
    assert outs[1].shape == (1, 192, 14, 18)
    assert outs[2].shape == (1, 384, 7, 9)
    assert outs[3].shape == (1, 768, 4, 5)
