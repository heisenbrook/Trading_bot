from tvDatafeed import TvDatafeed, Interval
import pandas as pd
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


tv = TvDatafeed(user, psw)

btcusdt = tv.get_hist(symbol='BTCUSDT', 
                      exchange='BINANCE', 
                      interval=Interval.in_daily, 
                      n_bars=10000,
                      extended_session=True)

btcusdt.drop('symbol', axis=1, inplace=True)

btcusdt['RSI'] = RSI(2, btcusdt)
btcusdt['Stoch_RSI'] = Stoch_RSI(2, btcusdt['RSI'])

btcusdt.to_csv('data.csv')


    