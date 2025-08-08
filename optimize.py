from tvDatafeed import TvDatafeed, Interval
import json
import os
import torch
import optuna
from optuna.pruners import MedianPruner
from utils.keys import user, psw, data_folder
from utils.data import preprocess
from utils.testing import objective

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tv = TvDatafeed(user, psw)

btcusdt  = tv.get_hist(symbol='BTCUSDT', 
                      exchange='BINANCE', 
                      interval=Interval.in_4_hour, 
                      n_bars=10000,
                      extended_session=True)

btcusdt = preprocess(btcusdt)

pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=15, interval_steps=5)
sampler = optuna.samplers.TPESampler(seed=42)

study = optuna.create_study(direction='minimize', 
                            sampler=sampler, 
                            pruner=pruner, 
                            study_name='BTC_Transf',
                            load_if_exists=True)

study.optimize(lambda trial: objective(trial, device, btcusdt), n_trials=200, show_progress_bar=True)

best_params = study.best_params
best_value = study.best_value

print(f'Best trial: {best_value}')
print(f'Best params: {best_params}')

with open(os.path.join(data_folder, 'best_params.json'), 'w') as f:
    json.dump(best_params, f, indent=4)



