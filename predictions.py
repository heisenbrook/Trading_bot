from tvDatafeed import TvDatafeed, Interval
import numpy as np
import torch
import os
import json
from utils.keys import user, psw, data_folder, generator
from utils.data import BTCDataset, preprocess


model = torch.jit.load(os.path.join(data_folder,'td_best_model.pt'))
model.to('cpu')
model.eval()

with open(os.path.join(data_folder, 'best_params.json'), 'r') as f:
    best_params = json.load(f)


tv = TvDatafeed(user, psw)

btcusdt = tv.get_hist(symbol='BTCUSDT', 
                      exchange='BINANCE', 
                      interval=Interval.in_4_hour, 
                      n_bars=10000,
                      extended_session=True)


btcusdt = preprocess(best_params['horizon'], btcusdt)
btcusdt = btcusdt.iloc[-(best_params['win_size'] + best_params['horizon']):]
btcusdt.to_csv(os.path.join(data_folder, 'btcusdt_4h_processed.csv'))

processed_data = BTCDataset(btcusdt,
                           win_size=best_params['win_size'], 
                           horizon=best_params['horizon'],
                           is_training=False)

data_loader = torch.utils.data.DataLoader(processed_data)

def make_predictions(data_loader):
    all_preds = []
    all_time = []
    with torch.no_grad():
        for data, _, ts in data_loader:
            preds = model(data) 
            all_preds.append(preds)
            all_time.append(ts)

    all_preds = np.concatenate(all_preds, axis=0)
    all_time = np.concatenate(all_time, axis=0)

    all_preds = all_preds.reshape(-1,4)
    all_time = all_time.reshape(-1)

    df = processed_data.denorm_pred(all_preds, all_time)

    return df

pred_df = make_predictions(data_loader)
pred_df.to_csv(os.path.join(data_folder, 'predictions.csv'))
print(f'Predictions saved to {os.path.join(data_folder, "predictions.csv")}')