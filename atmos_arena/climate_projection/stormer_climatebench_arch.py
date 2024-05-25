#Third Party
import torch
import torch.nn as nn
import numpy as np
from stormer_arch import Stormer
from atmos_utils.pos_embed import get_1d_sincos_pos_embed_from_grid


class StormerClimateBench(Stormer):
    def __init__(self, 
        in_img_size,
        in_variables,
        out_variables,
        time_history=10,
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
        )
        
        self.time_history = time_history
        self.freeze_encoder = freeze_encoder
        
        # used to aggregate multiple timesteps in the input
        self.time_pos_embed = nn.Parameter(torch.zeros(1, time_history, hidden_size), requires_grad=True)
        self.time_agg = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.time_query = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)
        
        # initialize time embedding
        time_pos_embed = get_1d_sincos_pos_embed_from_grid(self.time_pos_embed.shape[-1], np.arange(self.time_history))
        self.time_pos_embed.data.copy_(torch.from_numpy(time_pos_embed).float().unsqueeze(0))
        
        # # overwrite Stormer
        # # use a linear prediction head for this task
        self.head = nn.Linear(hidden_size, in_img_size[0]*in_img_size[1])

        if freeze_encoder:
            for name, p in self.blocks.named_parameters():
                name = name.lower()
                # we do not freeze the norm layers, as suggested by https://arxiv.org/abs/2103.05247
                if 'norm' in name:
                    continue
                else:
                    p.requires_grad_(False)

    def forward(self, x, time_interval, variables):
        # x: `[B, T, V, H, W]` shape.
        b, t, _, _, _ = x.shape
        x = x.flatten(0, 1)  # BxT, V, H, W
        
        x = self.embedding(x, variables) # BxT, L, D
        x = self.embed_norm_layer(x) # BxT, L, D

        time_interval_emb = self.t_embedder(time_interval).repeat_interleave(t, dim=0) # time_interval_emb is an artifact from the pretrained model
        for block in self.blocks:
            x = block(x, time_interval_emb)
            
        x = x.unflatten(0, sizes=(b, t)) # B, T, L, D
        x = x + self.time_pos_embed.unsqueeze(2)
        # global average pooling, also used in CNN-LSTM baseline in ClimateBench
        x = x.mean(-2) # B, T, D
        time_query = self.time_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.time_agg(time_query, x, x)  # B, 1, D
        
        x = self.head(x)
        x = x.reshape(-1, 1, self.in_img_size[0], self.in_img_size[1]) # B, 1, H, W
        
        return x
