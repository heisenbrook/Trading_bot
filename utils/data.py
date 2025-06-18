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

        self.win_size = win_size
        self.horizon = horizon

        self.data = features.join(labels)

        self.cols = ['high','low','open','close','RSI','next_high','next_low','next_open','next_close']
        self.feat_cols = ['high','low','open','close','RSI','next_high','next_low','next_open','next_close','volume']
        self.target_col = labels.columns.to_list()
        self.timestamps = self.data.index.values
        self.time_index = np.arange(len(self.data))

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cols', Pipeline([
                         ('log_returns', FunctionTransformer(func= lambda x: (np.log(x) + 1e-7) , inverse_func= lambda x: (np.exp(x) + 1e-7) , check_inverse=False)),
                         ('robust', RobustScaler()),
                         ('power', PowerTransformer())
                ]), self.cols),
                ('volume', RobustScaler(), ['volume'])
            ],
            remainder='passthrough'
        )

        self.data[self.feat_cols] = self.preprocessor.fit_transform(self.data[self.feat_cols])

        self.data.ffill(inplace=True)

    def __len__(self):
        return len(self.data) - self.win_size - self.horizon + 1

    def __getitem__(self, i):
        x = self.data[self.feat_cols].values[i:i+self.win_size]
        y = self.data[self.target_col].values[i+self.win_size:i+self.win_size + self.horizon]
        last_time_index = self.time_index[i+self.win_size:i+self.win_size+self.horizon]

        return torch.FloatTensor(x), torch.FloatTensor(y), last_time_index
    
    def denorm_pred(self, y, last_time_index):
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
                for _, step in reversed(transf.steps):
                    if hasattr(step, 'inverse_transform'):
                        y_slice = step.inverse_transform(y_slice)

            elif hasattr(transf, 'inverse_transform'):
                y_slice = transf.inverse_transform(y_slice)
            
            dummy[:, start:end] = y_slice

        df = pd.DataFrame(dummy, columns=self.feat_cols, index=self.timestamps[last_time_index])

        return df[self.target_col]


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
    label.drop(['volume','RSI'], axis=1, inplace=True)
    label = label.rename({'high':'next_high','low':'next_low','open':'next_open','close':'next_close'}, axis='columns')

    return label

def preprocess(data):
    data.drop('symbol', axis=1, inplace=True)

    data['RSI'] = RSI(14, data)

    data = data.iloc[15:]

    volume = data.pop('volume')
    data.insert(len(data.columns), 'volume', volume)

    return data





    











    