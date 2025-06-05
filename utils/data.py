from sklearn.preprocessing import RobustScaler, FunctionTransformer, PowerTransformer, StandardScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer
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
        self.cols = ['high','low','open','close','RSI','next_open','next_close']
        self.feat_cols = ['high','low','open','close','RSI','next_open','next_close', 'volume']
        self.target_col = labels.columns.to_list()
        self.win_size = win_size
        self.horizon = horizon

        # self.last_close = self.data['close'].iloc[-1]
        # self.last_open = self.data['open'].iloc[-1]

        # self.pipeline = Pipeline([
        #     ('log_returns', FunctionTransformer(calculate_log_returns)),
        #     ('robust', RobustScaler()),
        #     ('power', PowerTransformer(method='yeo-johnson', standardize=True))
        # ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cols', Pipeline([
                         ('log_returns', FunctionTransformer(func= np.log, inverse_func=np.exp, check_inverse=False)),
                         ('robust', RobustScaler()),
                         ('power', PowerTransformer())
                ]), self.cols),
                ('volume', RobustScaler(), ['volume'])
            ],
            remainder='passthrough'
        )


        self.data[self.feat_cols] = self.preprocessor.fit_transform(self.data[self.feat_cols])

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
        dummy[:, (-len(self.target_col)-1):-1] = y
        start = 0
        for name, transf, feat in self.preprocessor.transformers_:
            if name == 'remainder':
                continue
            end = start + len(feat)
            y_slice = dummy[:, start:end]
            if isinstance(transf, Pipeline):
                for step_name, step in reversed(transf.steps):
                    if hasattr(step, 'inverse_transform'):
                        y_slice = step.inverse_transform(y_slice)

            elif hasattr(transf, 'inverse_transform'):
                y_slice = transf.inverse_transform(y_slice)
            
            dummy[:, start:end] = y_slice

                     
        # denorm = self.pipeline.named_steps['power'].inverse_transform(dummy)
        # denorm = self.pipeline.named_steps['robust'].inverse_transform(denorm) 

        # assert np.allclose(self.last_close, denorm[-1, -1]), 'data not denormalized'

        return dummy[:, (-len(self.target_col)-1):-1]


def RSI(n_candles, data):
    delta = data['close'].diff()
    gain = delta.where(delta>0, 0)
    loss = delta.where(delta<0, 0)
    
    avg_gain = gain.rolling(n_candles).mean()
    avg_loss = loss.rolling(n_candles).mean()

    rs = avg_gain/avg_loss
    rsi = 100 - (100/(1+rs))
    
    return rsi

def create_labels(data):
    label = data.shift(-6)
    label.iloc[:-6]
    label.drop(['high','low','volume','RSI'], axis=1, inplace=True)
    label = label.rename({'open':'next_open','close':'next_close'}, axis='columns')

    return label

def preprocess(data):
    data.drop('symbol', axis=1, inplace=True)

    data['RSI'] = RSI(14, data)

    data = data.iloc[15:]

    volume = data.pop('volume')
    data.insert(len(data.columns), 'volume', volume)

    return data

# def calculate_log_returns(data):
#     return np.log(data)

# def invert_log(data, last_open, last_close, i):

#     if i == 0:
#         cum_ret = np.cumsum(data)
#         data = last_close * np.exp(cum_ret)
#     elif i == 1:
#         cum_ret = np.cumsum(data)
#         data = last_open * np.exp(cum_ret)

#     return data




    











    