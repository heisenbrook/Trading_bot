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

def preprocess(data):
    data.drop('symbol', axis=1, inplace=True)

    data['RSI'] = RSI(14, data)
    data['Stoch_RSI'] = Stoch_RSI(14, data['RSI'])
    #data['buy/sell'] = np.where(data['RSI'] + data['Stoch_RSI'] > 50, 'Sell', 'Buy')

    min_max_scaler = MinMaxScaler()
    prices = data[['open','high','low','close']]
    prices_min_max = min_max_scaler.fit_transform(prices)

    data['day_of_the_week'] = data.index.day % 7
    data = cyclic_enc(data, 'day_of_the_week', 7)

    data[['open','high','low','close']] = prices_min_max

    data.insert(0, 'day_of_the_week', data.pop('day_of_the_week'))
    data.insert(1, 'day_of_the_week_sin', data.pop('day_of_the_week_sin'))
    data.insert(2, 'day_of_the_week_cos', data.pop('day_of_the_week_cos'))

    


tv = TvDatafeed(user, psw)

btcusdt = tv.get_hist(symbol='BTCUSDT', 
                      exchange='BINANCE', 
                      interval=Interval.in_daily, 
                      n_bars=10000,
                      extended_session=True)

preprocess(btcusdt)

btcusdt.to_csv('data.csv')




    