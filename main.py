from tvDatafeed import TvDatafeed, Interval
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.dash_app import app
from torch.utils.data import DataLoader, random_split
from utils.model import FinanceTransf, DirectionalAccuracyLoss
from utils.keys import user, psw, data_folder
from utils.data import BTCDataset
from utils.train import train_test
from utils.testing import testing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tv = TvDatafeed(user, psw)

btcusdt = tv.get_hist(symbol='BTCUSDT', 
                      exchange='BINANCE', 
                      interval=Interval.in_4_hour, 
                      n_bars=10000,
                      extended_session=True)

full_data = BTCDataset(btcusdt)

train_data, test_data, eval_data = random_split(full_data, [0.7 , 0.2, 0.1])

batch_size = 64
train_loader = DataLoader(train_data, batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size, shuffle=False)
eval_loader = DataLoader(eval_data, batch_size, shuffle=False)

td_bot = FinanceTransf(
    num_features=len(full_data.feat_cols),
    n_targets=len(full_data.target_col),
    n_layers=4
)
td_bot.to(device)
for p in td_bot.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

criterion = DirectionalAccuracyLoss()
optimizer = optim.Adam(td_bot.parameters(), lr=0.0001, weight_decay=1e-5)
nn.utils.clip_grad_norm_(td_bot.parameters(), max_norm=1.0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

train_test(device, 500, td_bot, optimizer, criterion, scheduler, train_loader, test_loader)

td_bot = torch.jit.load(os.path.join(data_folder,'td_best_model.pt'))

testing(device, td_bot, eval_loader, full_data)


app.run(debug=True, use_reloader=False, port=8050)