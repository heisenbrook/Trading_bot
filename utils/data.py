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
    def __init__(self, features, win_size, horizon, is_training=True, preprocessor=None):

        self.win_size = win_size
        self.horizon = horizon
        self.data = features
        self.is_training = is_training

        if isinstance(self.data, pd.Series):
            self.data = self.data.to_frame()

        self.prices_col = ['high', 'low', 'open', 'close']
        self.momentum_col = ['RSI', 'MOM', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'ADX', 'ROC']
        self.bands_col = ['BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER', 'SMA_50', 'PLAW', 'PLAW_BANDS_LOW', 'PLAW_BANDS_UP']
        self.patterns_col = ['dist_nearest_support', 'dist_nearest_resistance', 'strength_support', 'strength_resistance']
        self.statistical_col = ['TSF', 'VAR', 'LINREG', 'STDDEV']
        self.volume_col = ['volume', 'OBV']
        self.target_col = ['next_close']
        self.feat_cols = self.data.columns.to_list()
        self.feat_cols_num = [len(self.prices_col), len(self.momentum_col), len(self.bands_col), len(self.patterns_col), len(self.statistical_col), len(self.volume_col)]
        self.feat_cols_tot = len(self.prices_col) + len(self.momentum_col) + len(self.bands_col) + len(self.patterns_col) + len(self.statistical_col) + len(self.volume_col) + len(self.target_col)
        self.timestamps = self.data.index.values
        self.time_index = np.arange(len(self.data))
        
        if self.is_training:
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('prices', Pipeline([
                             ('robust', RobustScaler()),
                             ('power', PowerTransformer(method='yeo-johnson', standardize=True))
                    ]), self.prices_col),
                    ('volume', Pipeline([
                             ('robust', RobustScaler()),
                             ('power', PowerTransformer(method='yeo-johnson', standardize=True))
                    ]), self.volume_col),
                    ('momentum', StandardScaler(), self.momentum_col),
                    ('patterns', StandardScaler(), self.patterns_col),
                    ('bands', MinMaxScaler(), self.bands_col),
                    ('statistical', Pipeline([
                             ('robust', RobustScaler()),
                             ('power', PowerTransformer(method='yeo-johnson', standardize=True))
                    ]), self.statistical_col),
                    ('targets', Pipeline([
                             ('robust', RobustScaler()),
                             ('power', PowerTransformer(method='yeo-johnson', standardize=True))
                    ]), self.target_col)
               ],
                remainder='passthrough'
              )
            self.data[self.feat_cols] = self.preprocessor.fit_transform(self.data[self.feat_cols])
        else:
            if preprocessor is None:
                raise ValueError("Preprocessor must be provided for non-training datasets.")
            self.preprocessor = preprocessor
            self.data[self.feat_cols] = self.preprocessor.transform(self.data[self.feat_cols])

        self.data = self.data.ffill()

    def __len__(self):
        return max(0, len(self.data) - self.win_size - self.horizon + 1)

    def __getitem__(self, i):
        x = self.data[self.feat_cols].values[i:i+self.win_size]
        y = self.data[self.target_col].values[i+self.win_size:i+self.win_size + self.horizon]
        last_time_index = i + self.win_size + self.horizon -1

        return torch.FloatTensor(x), torch.FloatTensor(y), last_time_index    
    
    def denorm_pred(self, y, last_time_index):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        
        num_targets = len(self.target_col)
        y_flat = y.reshape(-1, num_targets)
        
        
        transf = self.preprocessor.named_transformers_['targets']

        if isinstance(transf, Pipeline):
            for _, step in reversed(transf.steps):
                if hasattr(step, 'inverse_transform'):
                    y_flat = step.inverse_transform(y_flat)
        
        if isinstance(last_time_index, (list, np.ndarray)):
            all_dates = []
            for ts_idx in last_time_index:
                current_start_timestamp = self.timestamps[ts_idx]
                all_dates.append(pd.Series(pd.date_range(start=current_start_timestamp, periods=self.horizon, freq='4h')))
            dates = pd.concat(all_dates)
        else:
            start_timestamp = self.timestamps[last_time_index]
            dates = pd.date_range(start=start_timestamp, periods=self.horizon, freq='4h')

        df = pd.DataFrame(y_flat, columns=self.target_col, index=dates)

        return df

# =============================================
# Preprocessing functions
# =============================================

def preprocess(horizon, data: pd.DataFrame):
    """
    Main preprocessing function to prepare raw historical data for modeling.
    It computes technical indicators, fits power law trends, and adds support/resistance features.
    """
    if 'symbol' in data.columns:
        data.drop('symbol', axis=1, inplace=True)

    # Indicator calculations
    # =============================

    # Momentum indicators
    data['RSI'] = talib.RSI(data['close'], timeperiod=14)
    data['MOM'] = talib.MOM(data['close'], timeperiod=10)
    data['MACD'], data['MACD_SIGNAL'], data['MACD_HIST'] = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['ADX'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
    data['ROC'] = talib.ROC(data['close'], timeperiod=10)

    # Overlap indicators
    data['BBANDS_UPPER'], data['BBANDS_MIDDLE'], data['BBANDS_LOWER'] = talib.BBANDS(data['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data['SMA_50'] = talib.SMA(data['close'], timeperiod=50)
    data['PLAW'] = fit_power_law(data['close'])
    data['PLAW_BANDS_LOW'], data['PLAW_BANDS_UP'] = powerlaw_bands(data['PLAW'], data['close'])

    # Pattern recognition indicators
    data = add_res_support_features(data, 14)

    # Statitical indicators
    data['TSF'] = talib.TSF(data['close'], timeperiod=14)
    data['VAR'] = talib.VAR(data['close'], timeperiod=14, nbdev=1)
    data['LINREG'] = talib.LINEARREG(data['close'], timeperiod=14)
    data['STDDEV'] = talib.STDDEV(data['close'], timeperiod=14, nbdev=1)

    # Volume indicators
    data['OBV'] = talib.OBV(data['close'], data['volume'])

    # Create future price labels
    labels = create_labels(horizon, data)
    data = data.join(labels)

    # Remove rows with NaN values
    data = data.dropna()

    return data


def create_labels(horizon, data: pd.DataFrame):
    """
    Create future price labels for time series forecasting.
    The function shifts the price columns by the specified horizon to create labels for the next time steps.
    It also removes unnecessary columns to focus on the target variables.
    """
    label = data.shift(horizon)
    label = label.iloc[horizon:]
    label.drop(columns=['high', 'low', 'open', 'volume', 'RSI', 'MOM', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'ADX', 'ROC',
                        'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER', 'SMA_50', 'PLAW', 'PLAW_BANDS_LOW', 'PLAW_BANDS_UP','TSF',
                        'VAR', 'LINREG', 'STDDEV',
                        'dist_nearest_support', 'dist_nearest_resistance', 'strength_support', 'strength_resistance',
                        'OBV'], axis=1, inplace=True)
    label = label.rename({'close':'next_close'}, axis='columns')

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

    params, _ = curve_fit(power_law, x_fit, y_fit, p0=[1, 1], bounds=([0.001, -5], [np.inf, 5]))

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

    

    











    