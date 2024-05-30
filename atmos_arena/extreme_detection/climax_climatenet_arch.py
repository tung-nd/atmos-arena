# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import torch.nn as nn
import torch
from climax_arch import ClimaX


class ClimaXClimateNet(ClimaX):
    def __init__(
        self,
        default_vars,
        out_vars,
        img_size=[768, 1152],
        patch_size=8,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        freeze_encoder=False,
    ):
        assert out_vars is not None

        super().__init__(
            default_vars,
            img_size,
            patch_size,
            embed_dim,
            depth,
            decoder_depth,
            num_heads,
            mlp_ratio,
            drop_path,
            drop_rate,
        )

        self.out_vars = out_vars
        self.freeze_encoder = freeze_encoder

        # overwrite ClimaX
        # use a linear prediction head for this task
        self.head = nn.Linear(embed_dim, len(self.out_vars) * patch_size**2)

        if freeze_encoder:
            for name, p in self.blocks.named_parameters():
                name = name.lower()
                # we do not freeze the norm layers, as suggested by https://arxiv.org/abs/2103.05247
                if 'norm' in name:
                    continue
                else:
                    p.requires_grad_(False)
                    
    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = len(self.out_vars)
        h = self.in_img_size[0] // p if h is None else h // p
        w = self.in_img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, lead_times, variables, out_variables=None):
        x = self.forward_encoder(x, lead_times, variables)  # B, L, D
        preds = self.head(x)
        return self.unpatchify(preds)
