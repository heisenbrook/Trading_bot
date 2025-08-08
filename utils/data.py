from sklearn.preprocessing import RobustScaler, FunctionTransformer, PowerTransformer, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


class BTCDataset(Dataset):
    def __init__(self, features, win_size, horizon):

        self.win_size = win_size
        self.horizon = horizon

        labels = create_labels(features, self.horizon)

        self.data = features.join(labels)
        
        self.prices_col = ['high', 'low', 'open', 'close']
        self.bands_col = ['power_law_lower', 'power_law_upper', 'power_law_bands_lower', 'power_law_bands_upper']
        self.indicators_col = ['RSI', 'power_law']
        self.volume_col = ['volume']
        self.feat_cols = self.data.columns.to_list()
        self.target_col = labels.columns.to_list()
        self.timestamps = self.data.index.values
        self.time_index = np.arange(len(self.data))

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('prices', Pipeline([
                         ('robust', RobustScaler()),
                         ('power', PowerTransformer())
                ]), self.prices_col),
                ('volume', PowerTransformer(), self.volume_col),
                ('indicators', StandardScaler(), self.indicators_col),
                ('bands', MinMaxScaler(), self.bands_col),
                ('targets', Pipeline([
                         ('robust', RobustScaler()),
                         ('power', PowerTransformer())
                ]), self.target_col)
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
        
        dummy = np.zeros((len(y), len(self.prices_col)))
        dummy[:, -len(self.target_col):] = y
        
        transf = self.preprocessor.named_transformers_['targets']

        if isinstance(transf, Pipeline):
            for _, step in reversed(transf.steps):
                if hasattr(step, 'inverse_transform'):
                    dummy = step.inverse_transform(dummy)

        df = pd.DataFrame(dummy, columns=self.target_col, index=self.timestamps[last_time_index])

        return df
    
    def denorm_train(self, y):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        
        dummy = np.zeros((len(y), len(self.prices_col)))
        dummy[:, -len(self.target_col):] = y
        
        transf = self.preprocessor.named_transformers_['prices']

        if isinstance(transf, Pipeline):
            for _, step in reversed(transf.steps):
                if hasattr(step, 'inverse_transform'):
                    dummy = step.inverse_transform(dummy)



def RSI(n_candles, data):
    delta = data['close'].diff()
    gain = delta.where(delta>0, 0)
    loss = delta.where(delta<0, 0)
    
    avg_gain = gain.rolling(n_candles).mean()
    avg_loss = loss.rolling(n_candles).mean()

    rs = avg_gain/avg_loss
    rsi = 100 - (100/(1+rs))
    
    return rsi

def create_labels(data, horizon):
    label = data.shift(-horizon)
    label.iloc[:-horizon]
    label.drop(['RSI','power_law','power_law_lower','power_law_upper','power_law_bands_lower','power_law_bands_upper','volume'], axis=1, inplace=True)
    label = label.rename({'high':'next_high','low':'next_low','open':'next_open','close':'next_close'}, axis='columns')

    return label

def preprocess(data):
    data.drop('symbol', axis=1, inplace=True)

    data['RSI'] = RSI(14, data)
    data['power_law'] = fit_power_law(data['close'])
    data['power_law_lower'] = data['power_law'] * 0.5
    data['power_law_upper'] = data['power_law'] * 2.0
    data['power_law_bands_lower'], data['power_law_bands_upper'] = powerlaw_bands(data['power_law'], data['close'])


    data = data.iloc[15:]

    # volume = data.pop('volume')
    # data.insert(len(data.columns), 'volume', volume)

    return data

def power_law(x, a, b):
    return a * (x**b)

def fit_power_law(data):
    x = np.arange(len(data))
    y = data.values
    valid = y > 0
    x_fit = x[valid]
    y_fit = y[valid]

    params, _ = curve_fit(power_law, x_fit, y_fit, p0=[1, 1])

    fitted_y = power_law(x, *params)

    return fitted_y

def powerlaw_bands(pw, close, window = 1460):
    residual = (close - pw) / pw
    rolling_std = residual.rolling(window=window, min_periods=1).std()
    lower_band = pw * (1 - 1.5 * rolling_std)
    upper_band = pw * (1 + 1.5 * rolling_std) 
    return lower_band, upper_band




    











    