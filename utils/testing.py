import torch
import os
from utils.keys import data_folder
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import mean_absolute_error


def metrics(targets, preds):

    mae_open = mean_absolute_error(targets['next_open'], preds['next_open'])
    mae_close = mean_absolute_error(targets['next_close'], preds['next_close'])
    max_drawdown = np.max(np.abs(targets['next_close'] - preds['next_close']))

    return mae_open, mae_close, max_drawdown

def plot_bar(targets, preds):

    fig = make_subplots(rows=1, cols=2, column_titles=['Real candles','Predicted candles'])

    fig.add_trace(go.Candlestick(x = targets.index,
                                         high= targets['next_high'],
                                         low= targets['next_low'],
                                         open= targets['next_open'],
                                         close= targets['next_close']),
                                         row=1, col=1)

    fig.add_trace(go.Candlestick(x = preds.index,
                                         high= preds['next_high'],
                                         low= preds['next_low'],
                                         open= preds['next_open'],
                                         close= preds['next_close']),
                                         row=1, col=2)
    
    fig.update_layout(height=600, 
                      width=1200,
                      title='Real vs predictions',
                      xaxis1=dict(rangeslider=dict(visible=False)),
                      xaxis2=dict(rangeslider=dict(visible=False)))
    

    fig.write_image(os.path.join(data_folder,'Pred_vs_real_candles.png'))
   


def testing(device, model, loader, full_data):
    model.eval()
    all_preds, all_targets, all_time = [], [], []

    with torch.no_grad():
        for data, label, time in loader:
            data = data.to(device)
            preds = model(data) 
            all_preds.append(preds.cpu().numpy())
            all_targets.append(label.cpu().numpy())
            all_time.append(time)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_time = np.concatenate(all_time, axis=0)

    all_preds = all_preds.reshape(-1,4)
    all_targets = all_targets.reshape(-1,4)
    all_time = all_time.reshape(-1)

    preds_real = full_data.denorm_pred(all_preds, all_time)
    targets_real = full_data.denorm_pred(all_targets, all_time)

    preds_real.sort_index().to_csv(os.path.join(data_folder,'preds_real.csv'))
    targets_real.sort_index().to_csv(os.path.join(data_folder,'targets_real.csv'))

    plot_bar(targets_real, preds_real)

    m_open, m_close, max_drawdown = metrics(targets_real, preds_real)

    print(f'MAE Open: ${m_open:.2f}')
    print(f'MAE Close: ${m_close:.2f}')
    print(f'Max Drawdown: ${max_drawdown:.2f}')






