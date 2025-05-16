from sklearn.preprocessing import RobustScaler, FunctionTransformer, PowerTransformer, StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

class BTCDataset(Dataset):
    def __init__(self, features, win_size=24, horizon=6):
        features = preprocess(features)
        labels = create_labels(features)

        self.data = features.join(labels)
        self.data = self.data.reset_index(drop=True)
        self.feat_cols = ['high','low','open','close','next_open','next_close']
        self.target_col = labels.columns.to_list()
        self.win_size = win_size
        self.horizon = horizon

        self.last_close = self.data['close'].iloc[-1]
        self.last_open = self.data['open'].iloc[-1]

        self.pipeline = Pipeline([
            ('log_returns', FunctionTransformer(calculate_log_returns)),
            ('robust', RobustScaler(quantile_range=(5, 95))),
            ('power', PowerTransformer())
        ])


        self.vol_scaler = Pipeline([
            ('robust', RobustScaler(quantile_range=(5, 95))),
            ('power', PowerTransformer())
        ])

        self.data[['volume']] = self.vol_scaler.fit_transform(self.data[['volume']])

        self.data[self.feat_cols] = self.pipeline.fit_transform(self.data[self.feat_cols])

        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.dropna(axis=0, inplace=True)




    def __len__(self):
        return len(self.data) - self.win_size - self.horizon + 1

    def __getitem__(self, i):
        x = self.data[self.feat_cols].values[i:i+self.win_size]
        y = self.data[self.target_col].values[i+self.win_size:i+self.win_size + self.horizon]

        return torch.FloatTensor(x), torch.FloatTensor(y)
    
    def denorm_pred(self, y):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        dummy = np.zeros((y.shape[0], len(self.feat_cols)))
        dummy[:, :len(self.target_col)] = y
        denorm = self.pipeline.named_steps['power'].inverse_transform(dummy)
        denorm = self.pipeline.named_steps['robust'].inverse_transform(denorm)
        denorm = np.nan_to_num(denorm)
        
        col=0

        for i in range((len(self.feat_cols)-1), 0, -1):
            col += 1 
            if col == 1:
                denorm[:,i] = invert_log(denorm[:,i], self.last_close)
            elif col == 2:
                denorm[:,i] = invert_log(denorm[:,i], self.last_open)

        return denorm[:, :len(self.target_col)]


def RSI(n_candles, data):
    delta = data['close'].diff()
    gain = delta.where(delta>0, 0)
    loss = delta.where(delta<0, 0)
    
    avg_gain = gain.rolling(n_candles).mean()
    avg_loss = loss.rolling(n_candles).mean()

    rs = avg_gain/avg_loss
    rsi = 100 - (100/(1+rs))
    
    return rsi

# def Stoch_RSI(n_candles, rsi):
#     min_rsi = rsi.rolling(n_candles).min()
#     max_rsi = rsi.rolling(n_candles).max()
    
#     stoch_rsi = ((rsi - min_rsi) / (max_rsi - min_rsi)) * 100
    
#     return stoch_rsi

def create_labels(data):
    label = data.shift(-6)
    label.iloc[:-6]
    label.drop(['high','low','volume','RSI'], axis=1, inplace=True)
    label = label.rename({'open':'next_open','close':'next_close'}, axis='columns')

    return label

def preprocess(data):
    data.drop('symbol', axis=1, inplace=True)

    data['RSI'] = RSI(14, data)
    # data['Stoch_RSI'] = Stoch_RSI(14, data['RSI'])

    data = data.iloc[15:]

    return data

def calculate_log_returns(data):
    return np.log((data)/ (data.shift(1)))

def invert_log(data, last_price):
    cum_ret = np.cumsum(data)
    return last_price * np.exp(cum_ret)




    











    