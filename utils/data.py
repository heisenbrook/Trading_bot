from tvDatafeed import TvDatafeed, Interval
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from keys import user, psw

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

# def pos_enc(seq_len, d_model):
#     pos = np.arange(seq_len)[:, np.newaxis]
#     i = np.arange(d_model)[np.newaxis, :]
#     angle_rads = pos / np.power(10000, (2 * (i // 2)) / d_model)

#     pe = np.zeros((seq_len, d_model))
#     pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
#     pe[:, 1::2] = np.cos(angle_rads[:, 1::2])

#     return pe

def create_labels(data):
    label = data.shift(-1)
    label.iloc[:-1]
    label.drop(['high','low','volume','RSI','Stoch_RSI','log_returns'], axis=1, inplace=True)
    label = label.rename({'open':'next_open','close':'next_close'}, axis='columns')

    return label

def create_seq(features, win_size=24, horizon=6, target_col=['next_open','next_close']):
    labels = create_labels(features)
    data = features.join(labels)
    data = data.reset_index(drop=True)
    feat_cols = data.columns.to_list()

    x, y = [],[]
    for i in range(len(data) - win_size - horizon +1):
        x.append(data[feat_cols].values[i:i+win_size])
        y.append(data[target_col].values[i+win_size:i+win_size+horizon])

    return np.array(x), np.array(y)


def preprocess(data):
    data.drop('symbol', axis=1, inplace=True)

    data['RSI'] = RSI(14, data)
    data['Stoch_RSI'] = Stoch_RSI(14, data['RSI'])
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))

    min_max_scaler = MinMaxScaler()
    prices = data[['open','high','low','close','volume','RSI','Stoch_RSI']]
    prices_min_max = min_max_scaler.fit_transform(prices)
    data[['open','high','low','close','volume','RSI','Stoch_RSI']] = prices_min_max

    # data['day_of_the_month'] = data.index.day
    # data['interval'] = data.groupby(data['day_of_the_month']).cumcount() + 1

    # data.insert(0, 'day_of_the_month', data.pop('day_of_the_month'))
    # data.insert(1, 'interval', data.pop('interval'))
    # data.insert(2, 'interval_sin', data.pop('interval_sin'))
    # data.insert(3, 'interval_cos', data.pop('interval_cos'))

    data = data.iloc[27:]

    return data



    


tv = TvDatafeed(user, psw)

btcusdt = tv.get_hist(symbol='BTCUSDT', 
                      exchange='BINANCE', 
                      interval=Interval.in_4_hour, 
                      n_bars=10000,
                      extended_session=True)

btcusdt = preprocess(btcusdt)

x, y = create_seq(btcusdt)

print(x.shape, y.shape)






    