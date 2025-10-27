from tvDatafeed import TvDatafeed, Interval
import plotly.graph_objects as go
import numpy as np
import torch
import os
import json
from utils.keys import user, psw, data_folder, train_data_folder
from utils.data import BTCDataset, preprocess


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

def plot_predictions(btcusdt, preds_df):
    """
    Plots the newly predicted candles on the current data.
    """
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x = btcusdt.index,
                                         high= btcusdt['high'],
                                         low= btcusdt['low'],
                                         open= btcusdt['open'],
                                         close= btcusdt['close'],
                                         name='Historical candles'))

    fig.add_trace(go.Scatter(x = preds_df.index,
                                         y= preds_df['close'],
                                         mode='lines',
                                         name='Predicted closes',
                                         line=dict(color='red')))
    
    fig.add_trace(go.Scatter(x = preds_df.index.tolist() + pred_df.index.tolist()[::-1],
                                         y= preds_df['range_high'].tolist() + preds_df['range_low'].tolist()[::-1],
                                         fill='toself',
                                         fillcolor='rgba(255, 182, 193, 0.5)',
                                         mode='lines',
                                         name='Predicted ranges',
                                         line=dict(color='red', dash='dash')))
    
    fig.update_layout(height=600, 
                      width=800,
                      title='Predicted candles',
                      xaxis1=dict(rangeslider=dict(visible=False)))
    
    fig.write_image(os.path.join(data_folder,'New_predicted_candles.png'))




model = torch.jit.load(os.path.join(train_data_folder,'td_best_model.pt'))
model.to('cpu')
model.eval()

with open(os.path.join(train_data_folder, 'best_params.json'), 'r') as f:
    best_params = json.load(f)

with open(os.path.join(train_data_folder, 'mae_close.json'), 'r') as f:
    mae_close_dict = json.load(f)
    
mae_close = mae_close_dict['mae_close']


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


pred_df = make_predictions(data_loader, mae_close)
plot_predictions(btcusdt.iloc[-18:], pred_df)
pred_df.to_csv(os.path.join(data_folder, 'predictions.csv'))
print(f'Predictions saved to {os.path.join(data_folder, "predictions.csv")}')