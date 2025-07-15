import torch
import torch.nn as nn

class MultiBranchTransformer(nn.Module):
    def __init__(self, input_dim=4, d_model=64, nhead=4, num_layers=2,
                 bio_classes=20, lang_classes=2, device_classes=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.bio_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, bio_classes))
        self.lang_head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, lang_classes))
        self.device_head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, device_classes))

    def forward(self, x):
        B = x.size(0)
        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.transformer(x)
        cls_out = x[:, 0]
        return self.bio_head(cls_out), self.lang_head(cls_out), self.device_head(cls_out)
