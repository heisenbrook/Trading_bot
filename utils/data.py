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
    def __init__(self, features, win_size, horizon, is_training=True, is_classification=False, preprocessor=None):

        self.win_size = win_size
        self.horizon = horizon
        self.data = features
        self.is_training = is_training
        self.is_classification = is_classification

        if isinstance(self.data, pd.Series):
            self.data = self.data.to_frame()

        self.prices_col = ['high', 'low', 'open', 'close']
        self.momentum_col = ['RSI']
        self.volume_col = ['volume', 'OBV']
        if self.is_classification:
            self.target_col = ['target_class']
        else:
            self.target_col = ['next_close']
        self.feat_cols = self.data.columns.to_list()
        self.feat_cols_num = [len(self.prices_col), len(self.momentum_col), len(self.volume_col)]
        self.feat_cols_tot = len(self.prices_col) + len(self.momentum_col) + len(self.volume_col) + len(self.target_col)
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
               ],
                remainder='passthrough'
              )
            self.data.loc[:,self.feat_cols] = self.preprocessor.fit_transform(self.data[self.feat_cols])
        else:
            if preprocessor is None:
                raise ValueError("Preprocessor must be provided for non-training datasets.")
            self.preprocessor = preprocessor
            self.data.loc[:,self.feat_cols] = self.preprocessor.transform(self.data[self.feat_cols])

        self.data = self.data.ffill().bfill()


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

def preprocess(horizon, data: pd.DataFrame, is_inference=False, is_classification=False):
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

    # Volume indicators
    data['OBV'] = talib.OBV(data['close'], data['volume'])

    # Create future price labels
    if is_classification:
        labels = create_labels_classification(horizon, data)
    else:
        labels = create_labels(horizon, data)
    data = data.join(labels)

    # Remove rows with NaN values
    if not is_inference:
        data = data.dropna()
    else:
        data = data.dropna(subset=['RSI'])  # Drop rows with NaN in essential features only

    return data


def create_labels(horizon, data: pd.DataFrame):
    """
    Create log returns labels for time series forecasting.
    The function shifts the price columns by the specified horizon to create labels for the next time steps.
    It also removes unnecessary columns to focus on the target variables.
    """

    future_closes = data['close'].shift(-horizon)
    current_closes = data['close']

    label = np.log(future_closes / current_closes).to_frame(name='next_close')

    return label

def create_labels_classification(horizon, data: pd.DataFrame, threshold=0.001):
    """
    Create classification labels for time series forecasting.
    The function shifts the price columns by the specified horizon to create labels for the next time steps.
    It classifies the future price movement into three classes: up, down, or stable based on the threshold.
    1 -> up
    0 -> stable/down
    """

    future_closes = data['close'].shift(-horizon)
    current_closes = data['close']

    label = pd.Series(0, index=data.index)

    returns = future_closes / current_closes

    label[returns > (1 + threshold)] = 1 

    return label.to_frame(name='target_class')



    

    











    