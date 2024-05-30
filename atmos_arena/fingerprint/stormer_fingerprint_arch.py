import torch.nn as nn
from stormer_arch import Stormer


class StormerFingerprint(Stormer):
    def __init__(self, 
        in_img_size,
        in_variables,
        patch_size=2,
        embed_norm=True,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        freeze_encoder=True
    ):
        super().__init__(
            in_img_size,
            in_variables,
            None,
            patch_size,
            embed_norm,
            hidden_size,
            depth,
            num_heads,
            mlp_ratio,
            freeze_encoder=freeze_encoder
        )
        
        self.freeze_encoder = freeze_encoder
        
        # overwrite Stormer
        # use an MLP prediction head for this task
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        if freeze_encoder:
            for name, p in self.blocks.named_parameters():
                name = name.lower()
                # we do not freeze the norm layers, as suggested by https://arxiv.org/abs/2103.05247
                if 'norm' in name:
                    continue
                else:
                    p.requires_grad_(False)

    def forward(self, x, time_interval, variables, out_variables=None):
        # x: `[B, V, H, W]` shape.
        x = self.embedding(x, variables) # B, L, D
        x = self.embed_norm_layer(x) # B, L, D
        time_interval_emb = self.t_embedder(time_interval) # time_interval_emb is an artifact from the pretrained model
        for block in self.blocks:
            x = block(x, time_interval_emb) # B, L, D
        x = x.mean(dim=1) # B, D
        return self.head(x) # B, 1
