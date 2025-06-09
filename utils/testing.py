import torch
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error

def metrics(targets, preds):

    mae_open = mean_absolute_error(targets['next_open'], preds['next_open'])
    mae_close = mean_absolute_error(targets['next_close'], preds['next_close'])

    return mae_open, mae_close

def plot_one(targets, preds):

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(targets['next_open'], label='True Open')
    plt.plot(preds['next_open'], label='Pred Open')
    plt.title('Open Prices')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(targets['next_close'], label='True Close')
    plt.plot(preds['next_close'], label='Pred Close')
    plt.title('Close Prices')
    plt.legend()

    plt.tight_layout()
    plt.savefig('Pred_vs_real.png')
    plt.close()

def plot_bar(targets, preds, btcusdt):

    dates = pd.to_datetime(btcusdt.index, format='%Y-%m-%d %H:%M')

    fig = make_subplots(rows=1, cols=2)

    # fig.add_trace(go.Candlestick(x = dates,
    #                                      high= btcusdt['high'],
    #                                      low= btcusdt['low'],
    #                                      open= btcusdt['open'],
    #                                      close= btcusdt['close']),
    #                                      row=1, col=1)

    fig.add_trace(go.Candlestick(x = dates,
                                         high= targets['next_high'],
                                         low= targets['next_low'],
                                         open= targets['next_open'],
                                         close= targets['next_close']),
                                         row=1, col=1)

    fig.add_trace(go.Candlestick(x = dates,
                                         high= preds['next_high'],
                                         low= preds['next_low'],
                                         open= preds['next_open'],
                                         close= preds['next_close']),
                                         row=1, col=2)

    fig.write_image('Pred_vs_real_candles.png')
   


def testing(device, model, loader, full_data, btcusdt):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data, label in loader:
            data = data.to(device)
            preds = model(data) 
            all_preds.append(preds.cpu().numpy())
            all_targets.append(label.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    all_preds = all_preds.reshape(-1,4)
    all_targets = all_targets.reshape(-1,4)

    preds_real = full_data.denorm_pred(all_preds)
    targets_real = full_data.denorm_pred(all_targets)

    plot_one(targets_real, preds_real)
    plot_bar(targets_real, preds_real, btcusdt)

    m_open, m_close = metrics(targets_real, preds_real)

    print(f'MAE Open: ${m_open:.2f}')
    print(f'MAE Close: ${m_close:.2f}')






