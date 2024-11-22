#Third Party
import torch
import torch.nn as nn
import torch.nn.functional as F
from atmos_arena.stormer_arch import Stormer
from atmos_arena.extreme_detection.cgnet import ConvBNPReLU


class StormerClimateNet(Stormer):
    def __init__(self, 
        in_img_size,
        in_variables,
        out_variables,
        patch_size=2,
        embed_norm=True,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        freeze_encoder=True
    ):
        assert out_variables is not None
        super().__init__(
            in_img_size,
            in_variables,
            out_variables,
            patch_size,
            embed_norm,
            hidden_size,
            depth,
            num_heads,
            mlp_ratio,
            freeze_encoder=freeze_encoder
        )
        
        self.embedding = nn.Sequential(
            ConvBNPReLU(len(in_variables), 256, 3, 2),
            ConvBNPReLU(256, 256, 3, 1),
            ConvBNPReLU(256, 512, 3, 2),
            ConvBNPReLU(512, 512, 3, 1),
            ConvBNPReLU(512, 1024, 3, 2),
            ConvBNPReLU(1024, 1024, 3, 1),
        )

        # overwrite Stormer
        # use a linear prediction head for this task
        self.head = nn.Linear(hidden_size, len(out_variables))

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
        c = len(self.out_variables)
        h = self.in_img_size[0] // p if h is None else h // p
        w = self.in_img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        p = 1
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, time_interval, variables, out_variables=None):
        x = self.embedding(x) # B, 1024, 96, 144
        x = x.flatten(2).transpose(1, 2) # B, 96*144, 1024
        x = self.embed_norm_layer(x) # B, 96*144, 1024

        time_interval_emb = self.t_embedder(time_interval) # time_interval_emb is an artifact from the pretrained model
        for block in self.blocks:
            x = block(x, time_interval_emb)
        
        x = self.head(x)
        x = self.unpatchify(x)
        x = F.interpolate(x, self.in_img_size, mode='bilinear',align_corners = False)   #Upsample score map, factor=8
        return x
