import torch
import torch.nn as nn
import numpy as np


class PosEnc(nn.Module):
    """
    Positional Encoding module. 
    Injects information about the relative or absolute position of the tokens in the sequence.
    Source: https://arxiv.org/abs/1706.03762
    """
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

    
class FeatAwareEmb(nn.Module):
    """
    Feature-Aware Embedding module.
    Embeds each feature group into a common d_model dimensional space and applies attention weights to each group.
    Source: Adapted from https://arxiv.org/abs/2106.05208
    """
    def __init__(self, d_model, features_dims):
        super().__init__()
        self.d_model = d_model
        self.feature_dims = features_dims
        self.num_groups = len(features_dims)

        self.group_embs = nn.ModuleList([
            nn.Linear(dim, d_model) for dim in features_dims
            ])
        
        self.att_weights = nn.Parameter(torch.ones(self.num_groups))

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):

        start = 0
        group_emb = []
        for i, group_emb_layer in enumerate(self.group_embs):
            dim = self.feature_dims[i]
            group_data = x[:, :, start:start + dim]
            group_emb_tensor = group_emb_layer(group_data)
            group_emb.append(group_emb_tensor)
            start += dim
        
        weights = torch.softmax(self.att_weights, dim=0)
        group_emb = torch.sum(torch.stack([w * e for w, e in zip(weights, group_emb)]), dim=0)
        
        x = self.layer_norm(group_emb)
        x = self.dropout(x)

        return x
    

class FinanceTransf(nn.Module):
    """
    Transformer model for financial time series forecasting.
    Combines feature-aware embedding, positional encoding, and a standard Transformer encoder.
    """
    def __init__(self, 
                 num_features, 
                 n_targets, 
                 n_layers, 
                 d_model, 
                 n_heads,
                 dim_feedforward,
                 dropout,
                 activation,
                 horizon, 
                 win_size):
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon
 

        self.feat_emb = FeatAwareEmb(d_model, num_features)
        self.pe = PosEnc(d_model, win_size)
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=n_heads,
            dropout=dropout,
            activation=activation,
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

        x = self.feat_emb(x)
        x = x * torch.sqrt(torch.tensor(self.d_model, dtype=torch.uint8))
        x = self.pe(x)
        x = self.transformer(x)
        x = x[:, -self.horizon:, :]
        x = self.out(x)

        return x


class FinanceLSTM(nn.Module):
    """
    LSTM model for financial time series forecasting.
    """
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_layers, 
                 n_targets, 
                 dropout,
                 horizon):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, n_targets))
    
    def forward(self, x):

        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -self.horizon:, :]
        x = self.out(x)

        return x
    

class DirectionalAccuracyLoss(nn.Module):
    """
    Custom loss function that combines Mean Squared Error (MSE) with Directional Accuracy.
    The loss is a weighted sum of MSE and the proportion of correct directional predictions.    
    """
    def __init__(self, alpha: float | None, is_classification=False):
        super().__init__()
        self.alpha = alpha
        self.mse= nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.is_classification = is_classification

    def forward(self, preds, targets):

        if self.is_classification:
            return self.bce(preds, targets)
        
        else:
            custom_mse = self.mse(preds, targets)

            horizon_size = preds.shape[1]
            for i in range(horizon_size - 1):       
                pred_d = preds[:, i, :] - preds[:, i+1, :]
                target_d = targets[:, i, :] - targets[:, i+1, :]

            preds_s = torch.sign(pred_d)
            targets_s = torch.sign(target_d)
            correct = (preds_s == targets_s).float()

            return  self.alpha * custom_mse + (1 - self.alpha) * (1 - correct.mean())
    
    

