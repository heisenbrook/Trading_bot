from tvDatafeed import TvDatafeed, Interval
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import FinanceTransf
from keys import user, psw
from data import BTCDataset

tv = TvDatafeed(user, psw)

btcusdt = tv.get_hist(symbol='BTCUSDT', 
                      exchange='BINANCE', 
                      interval=Interval.in_4_hour, 
                      n_bars=10000,
                      extended_session=True)

full_data = BTCDataset(btcusdt)


train_size = int(0.8 + len(full_data))
test_size = len(full_data) - train_size

train_data, test_data = random_split(full_data, [train_size, test_size])

batch_size = 64
train_loader = DataLoader(train_data, batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size, shuffle=False)

model = FinanceTransf(
    num_features=len(full_data.feat_cols),
    n_targets=len(full_data.target_col),
    n_layers=2
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)