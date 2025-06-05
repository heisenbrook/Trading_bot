import torch
import torch.nn as nn
import numpy as np

class PosEnc(nn.Module):
    def __init__(self, d_model, win_size):
        super().__init__()
        pos = torch.arange(0, win_size, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        angle_rads = pos * div

        pe = torch.zeros(win_size, d_model)
        pe[:, 0::2] = torch.sin(angle_rads)
        pe[:, 1::2] = torch.cos(angle_rads)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1),:]
    

class FinanceTransf(nn.Module):
    def __init__(self, num_features, n_targets, n_layers, horizon=6, win_size=24, d_model=64):
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon

        self.input = nn.Linear(num_features, d_model)
        self.pe = PosEnc(d_model, win_size)
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=256,
            nhead=4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            layer_norm_eps=1e-6
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=n_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, n_targets))

    
    def forward(self, x):
        x = self.input(x)
        #x = x * np.sqrt(self.d_model)
        x = self.pe(x)
        x = self.transformer(x)
        x = x[:, -self.horizon:, :]

        return self.out(x)
