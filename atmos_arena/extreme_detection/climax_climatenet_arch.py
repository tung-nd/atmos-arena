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
import torch.nn.functional as F
from atmos_arena.climax_arch import ClimaX
from atmos_arena.extreme_detection.cgnet import ConvBNPReLU


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
            freeze_encoder=freeze_encoder
        )

        self.out_vars = out_vars
        self.freeze_encoder = freeze_encoder
        
        self.token_embeds = nn.Sequential(
            ConvBNPReLU(len(default_vars), 256, 3, 2),
            ConvBNPReLU(256, 256, 3, 1),
            ConvBNPReLU(256, 512, 3, 2),
            ConvBNPReLU(512, 512, 3, 1),
            ConvBNPReLU(512, 1024, 3, 2),
            ConvBNPReLU(1024, 1024, 3, 1),
        )

        # overwrite ClimaX
        # use a linear prediction head for this task
        self.head = nn.Linear(embed_dim, len(self.out_vars))

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

        p = 1
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, lead_times, variables, out_variables=None):
        x = self.token_embeds(x) # B, 1024, 96, 144
        x = x.flatten(2).transpose(1, 2) # B, 96*144, 1024
        
        # add pos embedding
        x = x + self.pos_embed

        # add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        lead_time_emb = lead_time_emb.unsqueeze(1)
        x = x + lead_time_emb  # B, L, D

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        preds = self.head(x)
        preds = self.unpatchify(preds)
        preds = F.interpolate(preds, self.in_img_size, mode='bilinear',align_corners = False)   #Upsample score map, factor=8
        return preds
