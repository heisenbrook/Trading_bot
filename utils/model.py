import torch
import torch.nn as nn
import numpy as np

class PosEnc(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        i = torch.arange(0, d_model, dtype=torch.float).unsqueeze(1)
        angle_rads = pos / np.power(10000, (2 * (i // 2)) / d_model)

        self.pe = torch.zeros(seq_len, d_model)
        self.pe[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        self.pe[:, 1::2] = torch.cos(angle_rads[:, 1::2])

    def forward(self, x):
        return x + self.pe[:x.size(1),:]
    

class FinanceTransf(nn.Module):
    def __init__(self, d_model, num_features, seq_len):
        super().__init__()
        self.input = nn.Linear(num_features, d_model)
        self.pe = PosEnc(seq_len, d_model)
        self.transformer = nn.Transformer(d_model)
        self.out = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.input(x)
        x = self.pe(x)
        x = self.transformer(x)

        return self.out(x)
