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

def cyclic_enc(data, col, period):
    data[f'{col}_sin'] = np.sin(2 * np.pi * data[col] / period)
    data[f'{col}_cos'] = np.cos(2 * np.pi * data[col] / period)
    return data

def create_seq(data, labels, seq_len, horizon=1):
    x, y = [],[]
    for i in range(len(data) - seq_len):
       if i == 0:
          continue
       x.append(data[i:i+seq_len])
       y.append(labels[i-1:])


def preprocess(data):
    data.drop('symbol', axis=1, inplace=True)

    data['RSI'] = RSI(14, data)
    data['Stoch_RSI'] = Stoch_RSI(14, data['RSI'])
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))

    min_max_scaler = MinMaxScaler()
    prices = data[['open','high','low','close','volume','RSI','Stoch_RSI']]
    prices_min_max = min_max_scaler.fit_transform(prices)
    data[['open','high','low','close','volume','RSI','Stoch_RSI']] = prices_min_max

    data['day_of_the_month'] = data.index.day
    data['interval'] = data.groupby(data['day_of_the_month']).cumcount() + 1
    data = cyclic_enc(data, 'interval', data['interval'].max())

    data.insert(0, 'day_of_the_month', data.pop('day_of_the_month'))
    data.insert(1, 'interval', data.pop('interval'))
    data.insert(2, 'interval_sin', data.pop('interval_sin'))
    data.insert(3, 'interval_cos', data.pop('interval_cos'))



    


tv = TvDatafeed(user, psw)

btcusdt = tv.get_hist(symbol='BTCUSDT', 
                      exchange='BINANCE', 
                      interval=Interval.in_4_hour, 
                      n_bars=10000,
                      extended_session=True)

preprocess(btcusdt)

btcusdt = btcusdt.iloc[27:]

labels = btcusdt.shift(-1)

btcusdt = btcusdt.iloc[:-1]
labels = labels.iloc[:-1]

btcusdt.to_csv('data.csv')
labels.to_csv('labels.csv')




    