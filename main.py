from tvDatafeed import TvDatafeed, Interval
from utils.keys import user, psw
from utils.data import BTCDataset

tv = TvDatafeed(user, psw)

btcusdt = tv.get_hist(symbol='BTCUSDT', 
                      exchange='BINANCE', 
                      interval=Interval.in_4_hour, 
                      n_bars=10000,
                      extended_session=True)

full_data = BTCDataset(btcusdt)

