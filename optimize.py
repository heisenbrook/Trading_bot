from tvDatafeed import Interval
import json
import os
import torch
import optuna
from optuna.pruners import MedianPruner
from utils.keys import tv, train_data_folder_tf, train_data_folder_lstm
from utils.testing import objective

#============================================
# Optimization script
#============================================

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
btcusdt  = tv.get_hist(symbol='BTCUSDT', 
                      exchange='BINANCE', 
                      interval=Interval.in_4_hour, 
                      n_bars=10000,
                      extended_session=True)

# Set up Optuna study
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=15, interval_steps=5)
sampler = optuna.samplers.TPESampler(seed=42)

input_model = input(f'Select model to optimize (tf/lstm): ')

study = optuna.create_study(direction='minimize', 
                            sampler=sampler, 
                            pruner=pruner, 
                            study_name='BTC_Transf',
                            load_if_exists=True)

# Run optimization
if input_model == 'tf':
    study.optimize(lambda trial: objective(trial, device, btcusdt), n_trials=200)
    folder = train_data_folder_tf
else:
    study.optimize(lambda trial: objective(trial, device, btcusdt, lstm=True), n_trials=200)
    folder = train_data_folder_lstm

best_params = study.best_params
best_value = study.best_value

print(f'Best trial: {best_value}')
print(f'Best params: {best_params}')

# Save the best parameters to a JSON file
with open(os.path.join(folder, 'best_params.json'), 'w') as f:
    json.dump(best_params, f, indent=4)






