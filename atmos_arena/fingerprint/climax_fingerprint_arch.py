import torch.nn as nn
from climax_arch import ClimaX


class ClimaXFingerprint(ClimaX):
    def __init__(
        self,
        default_vars,
        img_size=[64, 128],
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        freeze_encoder=False,
    ):

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
        self.freeze_encoder = freeze_encoder

        # overwrite ClimaX
        # use an MLP prediction head for this task
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        if freeze_encoder:
            for name, p in self.blocks.named_parameters():
                name = name.lower()
                # we do not freeze the norm layers, as suggested by https://arxiv.org/abs/2103.05247
                if 'norm' in name:
                    continue
                else:
                    p.requires_grad_(False)

    def forward(self, x, lead_times, variables):
        out_transformers = self.forward_encoder(x, lead_times, variables)  # B, L, D
        out_transformers = out_transformers.mean(dim=1) # B, D
        preds = self.head(out_transformers) # B, 1
        return preds
