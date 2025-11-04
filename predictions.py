from tvDatafeed import Interval
import plotly.graph_objects as go
import numpy as np
import torch
import os
import json
from utils.keys import tv, data_folder, train_data_folder, fine_tuning_data_folder
from utils.data import BTCDataset, preprocess
from utils.plotting import plot_predictions


def make_predictions(data_loader, mae_close):
    """
    Makes predictions using the trained model and returns a DataFrame with denormalized predictions."""
    all_preds = []
    all_time = []
    with torch.no_grad():
        for data, _, ts in data_loader:
            preds = model(data) 
            all_preds.append(preds)
            all_time.append(ts)

    all_preds = np.concatenate(all_preds, axis=0)
    all_time = np.concatenate(all_time, axis=0)

    all_preds = all_preds.reshape(-1)
    all_time = all_time.reshape(-1)

    df = processed_data.denorm_pred(all_preds, all_time)
    df['next_open'] = df['next_close'].shift(1)
    df['range_low'] = df['next_close'] - mae_close
    df['range_high'] = df['next_close'] + mae_close
    df = df.rename(columns={'next_open':'open','next_close':'close'})
    df = df.loc[:, ['range_low','open','close','range_high']]
    df.dropna(inplace=True)

    return df


if os.path.exists(os.path.join(fine_tuning_data_folder, f'td_finetuned_model.pt')):
    print('Loading fine-tuned model for predictions...')
    model_path = os.path.join(fine_tuning_data_folder,'td_finetuned_model.pt')
else:
    print('Loading base model for predictions...')
    model_path = os.path.join(train_data_folder,'td_best_model.pt')
model = torch.jit.load(model_path)
model.to('cpu')
model.eval()

with open(os.path.join(train_data_folder, 'best_params.json'), 'r') as f:
    best_params = json.load(f)

with open(os.path.join(fine_tuning_data_folder, 'continual_learning_log.json'), 'r') as f:
    results_dict = json.load(f)

last_entry = results_dict[-1]
if last_entry['mae_close'] == '-':
    last_entry = results_dict[-2]

mae_close = last_entry['mae_close']


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


pred_df = make_predictions(data_loader, mae_close)
plot_predictions(btcusdt.iloc[-18:], pred_df)
pred_df.to_csv(os.path.join(data_folder, 'predictions.csv'))
print(f'Predictions saved to {os.path.join(data_folder, "predictions.csv")}')