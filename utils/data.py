from sklearn.preprocessing import MinMaxScaler
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
        self.data.dropna(axis=0, inplace=True)
        self.feat_cols = features.columns.to_list()
        self.target_col = labels.columns.to_list()

        self.scaler = MinMaxScaler()
        self.data[self.feat_cols] = self.scaler.fit_transform(self.data[self.feat_cols])

        self.win_size = win_size
        self.horizon = horizon
    
    def __len__(self):
        return len(self.data) - self.win_size - self.horizon + 1

    def __getitem__(self, i):
        x = self.data[self.feat_cols].values[i:i+self.win_size]
        y = self.data[self.target_col].values[i+self.win_size:i+self.win_size + self.horizon]

        return torch.FloatTensor(x), torch.FloatTensor(y)
    
    def denorm_pred(self, y):
        dummy = np.zeros((len(y), len(self.feat_cols)))
        dummy[:,:len(self.target_col)] = y
        dummy[:,:len(self.target_col)] = denorm_log(self.data['open','close'], dummy[:,:len(self.target_col)])

        return self.scaler.inverse_transform(dummy)[:,:len(self.target_col)]


def RSI(n_candles, data):
    delta = data['close'].diff()
    gain = delta.where(delta>0, 0)
    loss = delta.where(delta<0, 0)
    
    avg_gain = gain.rolling(n_candles).mean()
    avg_loss = loss.rolling(n_candles).mean()

    rs = avg_gain/avg_loss
    rsi = 100 - (100/(1+rs))
    
    return rsi

def Stoch_RSI(n_candles, rsi):
    min_rsi = rsi.rolling(n_candles).min()
    max_rsi = rsi.rolling(n_candles).max()
    
    stoch_rsi = ((rsi - min_rsi) / (max_rsi - min_rsi)) * 100
    
    return stoch_rsi

def create_labels(data):
    label = data.shift(-6)
    label.iloc[:-6]
    label.drop(['high','low','volume','RSI','Stoch_RSI'], axis=1, inplace=True)
    label = label.rename({'open':'next_open','close':'next_close'}, axis='columns')

    return label

def preprocess(data):
    data.drop('symbol', axis=1, inplace=True)

    data['RSI'] = RSI(14, data)
    data['Stoch_RSI'] = Stoch_RSI(14, data['RSI'])
    data['close'] = np.log(data['close'] / data['close'].shift(1))
    data['open'] = np.log(data['open'] / data['open'].shift(1))


    data = data.iloc[27:]

    return data

def denorm_log(data, pred):
    cum_returns = np.cumsum(pred, axis=1)
    prices = data['close'].iloc(-1) * np.exp(cum_returns)

    return prices




    











    