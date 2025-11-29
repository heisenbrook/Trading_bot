from tvDatafeed import Interval
import numpy as np
import joblib
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.dash_app import app
from torch.utils.data import DataLoader
from utils.model import FinanceLSTM, FinanceTransf, DirectionalAccuracyLoss
from utils.keys import get_candles
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

# Function to choose model and load pre-trained weights

btcusdt, timeframe, input_model, input_class, train_data_folder = get_candles(10000)

# Load best hyperparameters and prepare datasets
# Preprocess data
# Save preprocessor
# Divide data into train, test and evaluation sets
# Create DataLoaders for each set
# Initialize model, weights, loss function, optimizer and learning rate scheduler

with open(os.path.join(train_data_folder, f'best_params_{timeframe}.json'), 'r') as f:
    best_params = json.load(f)


btcusdt = preprocess(best_params['horizon'], input_class, btcusdt)


n = len(btcusdt)
if n < best_params['win_size'] + best_params['horizon']:
    raise ValueError(f'Dataset too small for the given win_size and horizon.')
    
n_train = int(n * 0.7)
n_test = int(n * 0.2)
n_eval = n - n_train - n_test

if n_train < best_params['win_size'] + best_params['horizon'] or \
n_test < best_params['win_size'] + best_params['horizon'] or \
n_eval < best_params['win_size'] + best_params['horizon']:
    raise ValueError(f'Dataset split too small for the given win_size and horizon.')
    
train_df = btcusdt.iloc[:n_train]
test_df = btcusdt.iloc[n_train - best_params['win_size']:n_train + n_test]
eval_df = btcusdt.iloc[n_train + n_test - best_params['win_size']:]


train_data = BTCDataset(train_df, 
                        win_size=best_params['win_size'], 
                        horizon=best_params['horizon'],
                        input_class=input_class,
                        is_training=True)
    
# Save the preprocessor
preprocessor_path = os.path.join(train_data_folder, f'preprocessor_{timeframe}.pkl')
joblib.dump(train_data.preprocessor, preprocessor_path)
print(f'Preprocessor saved to {preprocessor_path}')
        
train_preprocessor = train_data.preprocessor

test_data = BTCDataset(test_df, 
                        win_size=best_params['win_size'], 
                        horizon=best_params['horizon'],
                        input_class=input_class,
                        is_training=False,
                        preprocessor=train_preprocessor)
        
eval_data = BTCDataset(eval_df, 
                        win_size=best_params['win_size'], 
                        horizon=best_params['horizon'],
                        input_class=input_class,
                        is_training=False,
                        preprocessor=train_preprocessor)






batch_size = best_params['batch_size']
train_loader = DataLoader(train_data, batch_size, num_workers=4, pin_memory=True, shuffle=True)
test_loader = DataLoader(test_data, batch_size, num_workers=4, pin_memory=True, shuffle=False)
eval_loader = DataLoader(eval_data, batch_size, num_workers=4, pin_memory=True, shuffle=False)

lr = best_params['lr']
alpha = best_params['alpha']

# Initialize model
if input_model == 'tf':
    td_bot = FinanceTransf(
        num_features=train_data.feat_cols_num,   
        n_targets=len(train_data.target_col),
        n_layers=best_params['n_layers'],
        d_model=best_params['d_model'],
        n_heads=best_params['n_heads'],
        dim_feedforward=best_params['dim_feedforward'],
        dropout=best_params['dropout'],
        activation=best_params['activation'],
        horizon=best_params['horizon'],
        win_size=best_params['win_size']
        
    )
else:
    td_bot = FinanceLSTM(
        input_size=train_data.feat_cols_tot,   
        n_targets=len(train_data.target_col),
        num_layers=best_params['num_layers'],
        hidden_size=best_params['hidden_size'],
        dropout=best_params['dropout'],
        horizon=best_params['horizon']
    )

td_bot.to(device)
for p in td_bot.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

weights = torch.tensor([1.5]).to(device)
criterion = DirectionalAccuracyLoss(alpha, weights, input_class)


if best_params['optim'] == 'adam':
    optimizer = torch.optim.Adam(td_bot.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
else:
    optimizer = torch.optim.SGD(td_bot.parameters(), lr=best_params['lr'], momentum=0.9, weight_decay=best_params['weight_decay'])


nn.utils.clip_grad_norm_(td_bot.parameters(), max_norm=1.0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, mode='min', factor=0.5)

# Train and evaluate the model
# Load the best model and test on evaluation set

train_test(device, best_params['n_epochs'], td_bot, optimizer, criterion, scheduler, train_loader, test_loader, timeframe, train_data_folder)

td_bot = torch.jit.load(os.path.join(train_data_folder,f'td_best_model_{timeframe}.pt'))

metrics_dict = testing(device, td_bot, eval_loader, eval_data, timeframe, input_class, train_data_folder)
    
for key, value in metrics_dict.items():
    if isinstance(value, np.float32):
        metrics_dict[key] = float(value)
    if isinstance(value, np.ndarray):
        metrics_dict[key] = '-'

with open(os.path.join(train_data_folder, f'metrics_{timeframe}.json'), 'w') as f:
    json.dump(metrics_dict, f, indent=4)


# Uncomment to run the Dash app for visualization
# app.run(debug=True, use_reloader=False, port=8050)