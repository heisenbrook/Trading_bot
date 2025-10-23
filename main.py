from tvDatafeed import TvDatafeed, Interval
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.dash_app import app
from torch.utils.data import DataLoader, random_split
from utils.model import FinanceTransf, DirectionalAccuracyLoss
from utils.keys import user, psw, train_data_folder, generator
from utils.data import BTCDataset, preprocess
from utils.train import train_test
from utils.testing import testing

#=======================================================================
# Main script to fetch data, train model, evaluate and run Dash app
#=======================================================================

# Set device and get data
# TvDatafeed can use your TradingView credentials to fetch data
# Source: https://github.com/rongardF/tvdatafeed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tv = TvDatafeed(user, psw)

btcusdt = tv.get_hist(symbol='BTCUSDT', 
                      exchange='BINANCE', 
                      interval=Interval.in_4_hour, 
                      n_bars=10000,
                      extended_session=True)


# Load best hyperparameters and prepare datasets
# Preprocess data
# Divide data into train, test and evaluation sets
# Create DataLoaders for each set
# Initialize model, weights, loss function, optimizer and learning rate scheduler

with open(os.path.join(train_data_folder, 'best_params.json'), 'r') as f:
    best_params = json.load(f)

btcusdt = preprocess(best_params['horizon'], btcusdt)

full_data = BTCDataset(btcusdt,
                       win_size=best_params['win_size'], 
                       horizon=best_params['horizon'])

train_data, test_data, eval_data = random_split(full_data, [0.7 , 0.2, 0.1], generator=generator)

batch_size = best_params['batch_size']
train_loader = DataLoader(train_data, batch_size, num_workers=4, pin_memory=True, shuffle=False)
test_loader = DataLoader(test_data, batch_size, num_workers=4, pin_memory=True, shuffle=False)
eval_loader = DataLoader(eval_data, batch_size, num_workers=4, pin_memory=True, shuffle=False)

lr = best_params['lr']
alpha = best_params['alpha']

td_bot = FinanceTransf(
    num_features=full_data.feat_cols_num,
    n_targets=len(full_data.target_col),
    n_layers=best_params['n_layers'],
    d_model=best_params['d_model'],
    n_heads=best_params['n_heads'],
    dim_feedforward=best_params['dim_feedforward'],
    dropout=best_params['dropout'],
    activation=best_params['activation'],
    horizon=best_params['horizon'], 
    win_size=best_params['win_size'] 
)

td_bot.to(device)
for p in td_bot.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

criterion = DirectionalAccuracyLoss(alpha)
optimizer = optim.Adam(td_bot.parameters(), lr=lr, weight_decay=1e-5)
nn.utils.clip_grad_norm_(td_bot.parameters(), max_norm=1.0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, mode='min', factor=0.5)

# Train and evaluate the model
# Load the best model and test on evaluation set

train_test(device, best_params['n_epochs'], td_bot, optimizer, criterion, scheduler, train_loader, test_loader)

td_bot = torch.jit.load(os.path.join(train_data_folder,'td_best_model.pt'))

mae_close, max_drawdown = testing(device, td_bot, eval_loader, full_data)

print(f'MAE Close: ${mae_close:.2f}')
print(f'Max Drawdown: ${max_drawdown:.2f}')

# Uncomment to run the Dash app for visualization
# app.run(debug=True, use_reloader=False, port=8050)