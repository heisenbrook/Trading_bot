from sklearn.preprocessing import RobustScaler, PowerTransformer, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset
import talib
import torch
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# =============================================
# Custom Dataset class for Bitcoin data
# =============================================

class BTCDataset(Dataset):
    """
    Custom Dataset for loading and preprocessing Bitcoin historical data.
    It handles feature engineering, normalization, and sequence generation for time series forecasting.  
    It also provides a method for denormalizing predictions back to the original scale.   
    """
    def __init__(self, features, win_size, horizon):

        self.win_size = win_size
        self.horizon = horizon

        labels = create_labels(features, self.horizon)

        self.data = features.join(labels)
        
        self.prices_col = ['high', 'low', 'open', 'close']
        self.bands_col = ['power_law_lower', 'power_law_upper', 'power_law_bands_lower', 'power_law_bands_upper']
        self.indicators_col = ['RSI','bbands_upper','bbands_middle','bbands_lower','sma_close','ema_close','sstar', 'power_law', 'dist_nearest_support', 'dist_nearest_resistance', 'strength_support', 'strength_resistance']
        self.volume_col = ['volume']
        self.feat_cols = self.data.columns.to_list()
        self.feat_cols_num = [len(self.prices_col), len(self.volume_col), len(self.indicators_col), len(self.bands_col)]
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

# =============================================
# Preprocessing functions
# =============================================

def preprocess(data):
    """
    Main preprocessing function to prepare raw historical data for modeling.
    It computes technical indicators, fits power law trends, and adds support/resistance features.
    """
    data.drop('symbol', axis=1, inplace=True)

    data['RSI'] = talib.RSI(data['close'], timeperiod=14)
    data['bbands_upper'], data['bbands_middle'], data['bbands_lower'] = talib.BBANDS(data['close'], timeperiod=20)
    data['sma_close'] = talib.SMA(data['close'], timeperiod=50)
    data['ema_close'] = talib.EMA(data['close'], timeperiod=50)
    data['sstar'] = talib.CDLSHOOTINGSTAR(data['open'], data['high'], data['low'], data['close'])
    data['power_law'] = fit_power_law(data['close'])
    data['power_law_lower'] = data['power_law'] * 0.5
    data['power_law_upper'] = data['power_law'] * 2.0
    data['power_law_bands_lower'], data['power_law_bands_upper'] = powerlaw_bands(data['power_law'], data['close'])
    data = add_res_support_features(data, 14)

    data = data.iloc[15:]

    return data


def create_labels(data, horizon):
    """
    Create future price labels for time series forecasting.
    The function shifts the price columns by the specified horizon to create labels for the next time steps.
    It also removes unnecessary columns to focus on the target variables.
    """
    label = data.shift(-horizon)
    label.iloc[:-horizon]
    label.drop(['RSI','bbands_upper','bbands_middle','bbands_lower','sma_close','ema_close','sstar','power_law','power_law_lower','power_law_upper','power_law_bands_lower','power_law_bands_upper','volume','dist_nearest_support', 'dist_nearest_resistance', 'strength_support', 'strength_resistance'], axis=1, inplace=True)
    label = label.rename({'high':'next_high','low':'next_low','open':'next_open','close':'next_close'}, axis='columns')

    return label

# =============================================
# Feature engineering functions
# =============================================

def power_law(x, a, b):
    """
    Power law function for curve fitting.
    Used to model relationships in financial time series data.
    See https://en.wikipedia.org/wiki/Power_law for more details.
    Inspired by Giovanni Santostasi's work:
    """
    return a * ((x + np.finfo(float).eps)**b)

def fit_power_law(data):
    """
    Fit a power law to the given data series.
    """
    x = np.arange(len(data))
    y = data.values
    valid = y > 0
    x_fit = x[valid]
    y_fit = y[valid]

    params, _ = curve_fit(power_law, x_fit, y_fit, p0=[1, 1])

    fitted_y = power_law(x, *params)

    return fitted_y

def powerlaw_bands(pw, close, window = 1460):
    """
    Calculate dynamic bands around a power law trend based on rolling standard deviation of residuals.
    """
    residual = (close - pw) / pw
    rolling_std = residual.rolling(window=window, min_periods=1).std()
    lower_band = pw * (1 - 1.5 * rolling_std)
    upper_band = pw * (1 + 1.5 * rolling_std) 
    return lower_band, upper_band


def nearest_distance(price, levels):
    """
    Calculate the distance to the nearest support or resistance level.  
    """
    if len(levels) == 0:
        return np.nan
    return np.min(np.abs(price - levels))

def strength_distance(price, levels, tolerance=0.01):
    """
    Calculate the strength of support or resistance based on proximity to multiple levels.    
    """
    if len(levels) == 0:
        return np.nan
    distances = np.abs(price - levels)
    strengths = 1 / (distances + tolerance)
    return np.sum(strengths)

def add_res_support_features(data, window):
    """
    Identify local support and resistance levels and compute features based on their proximity.
    """

    data = data.copy()

    data['local_min'] = data['low'].rolling(window, center=True).min()
    data['local_max'] = data['high'].rolling(window, center=True).max()

    data['is_support'] = (data['low'] == data['local_min'])
    data['is_resistance'] = (data['high'] == data['local_max'])

    supports = data[data['is_support']]['low'].values
    resistances = data[data['is_resistance']]['high'].values

    data['dist_nearest_support'] = data.apply(lambda row: nearest_distance(row['close'], supports), axis=1)
    data['dist_nearest_resistance'] = data.apply(lambda row: nearest_distance(row['close'], resistances), axis=1)

    data['strength_support'] = data.apply(lambda row: strength_distance(row['close'], supports), axis=1)
    data['strength_resistance'] = data.apply(lambda row: strength_distance(row['close'], resistances), axis=1)

    return data.drop(columns=['local_min', 'local_max', 'is_support', 'is_resistance'])

    

    











    