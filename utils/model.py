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
    def __init__(self, num_features, n_targets, n_layers, horizon, win_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon

        self.input = nn.Linear(num_features, d_model)
        self.pe = PosEnc(d_model, win_size)
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=4*d_model,
            nhead=max(2, d_model // 16),
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
        x = x * torch.sqrt(torch.tensor(self.d_model, dtype=torch.uint8))
        x = self.pe(x)
        x = self.transformer(x)
        x = x[:, -self.horizon:, :]

        return self.out(x)
    

class DirectionalAccuracyLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse= nn.L1Loss(reduction='mean')

    def forward(self, preds, targets):

        custom_mse = self.mse(preds, targets)

        preds_s = torch.sign(preds[:, :, -1] - preds[:, :, -2])
        targets_s = torch.sign(targets[:, :, -1] - targets[:, :, -2])
        correct = (preds_s == targets_s).float()

        return  self.alpha * custom_mse + (1 - self.alpha) * (1 - correct.mean())
    
    

