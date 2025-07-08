import torch
from torch.utils.data import DataLoader, random_split
from datetime import datetime as dt
import os
from utils.data import BTCDataset
from utils.keys import data_folder
from utils.train import train_epoch, eval_epoch
from utils.model import FinanceTransf, DirectionalAccuracyLoss
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import mean_absolute_error


def objective(trial, device, btcusdt):

    params = {
        'n_layers': trial.suggest_int('n_layers', 1, 4),
        'd_model': trial.suggest_int('d_model', 32, 128, step=16),
        'n_heads': trial.suggest_int('n_heads', 4, 8, step=4),
        'dim_feedforward': trial.suggest_int('dim_feedforward', 64, 512, step=64),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.1),
        'activation': trial.suggest_categorical('activation', ['relu', 'gelu']),
        'horizon': trial.suggest_int('horizon', 6, 24, step=6),
        'win_size': trial.suggest_int('win_size', 64, 256),
        'batch_size': trial.suggest_int('batch_size', 32, 128, step=16),
        'alpha': trial.suggest_float('alpha', 0.1, 0.9, step=0.1),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'n_epochs': trial.suggest_int('n_epochs', 50, 200, step=10)
    }


    full_data = BTCDataset(btcusdt, params['win_size'], params['horizon'])

    train_data, test_data, eval_data = random_split(full_data, [0.7 , 0.2, 0.1])

    train_loader = DataLoader(train_data, params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_data, params['batch_size'], shuffle=False)
    eval_loader = DataLoader(eval_data, params['batch_size'], shuffle=False)

    model = FinanceTransf(num_features=len(full_data.feat_cols),
                          n_targets=len(full_data.target_col),
                          n_layers=params['n_layers'],
                          d_model=params['d_model'],
                          n_heads=params['n_heads'],
                          dim_feedforward=params['dim_feedforward'],
                          dropout=params['dropout'],
                          activation=params['activation'],
                          horizon=params['horizon'], 
                          win_size=params['win_size'] 
)

    model.to(device)

    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    criterion = DirectionalAccuracyLoss(params['alpha'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-5)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, mode='min', factor=0.5)

    best_test_loss = float('inf')

    train_losses, test_losses = [], []
    patience = 0

    for epoch in range(params['n_epochs']):
        model.train()
        train_loss = train_epoch(device, epoch, params['n_epochs'], model, optimizer, criterion, train_loader)
        test_loss = eval_epoch(device, epoch, params['n_epochs'], model, criterion, test_loader)
    
        scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if (epoch + 1) % (params['n_epochs']//10) == 0 or (epoch + 1) == 1:
            x = dt.now()
            print(f'{x.strftime('%Y-%m-%d %H:%M:%S')}| Epoch {epoch + 1} | training loss:{train_loss:.5f}% | test loss:{test_loss:.5f}%')

        if test_loss < best_test_loss:
            patience = 0
            best_test_loss = test_loss
        elif epoch > 10 and test_loss > best_test_loss:
            patience += 1

        if epoch >30 and patience > 30:
            x = dt.now()
            print(f'{x.strftime('%Y-%m-%d %H:%M:%S')}| Epoch {epoch + 1} | training loss:{train_loss:.5f}% | test loss:{test_loss:.5f}%')
            print('Early stop')
            break

    model.eval()
    all_preds, all_targets, all_time = [], [], []

    with torch.no_grad():
        for data, label, time in eval_loader:
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
    m_open, m_close, max_drawdown = metrics(targets_real, preds_real)

    tot_loss = m_open + m_close + max_drawdown

    return tot_loss


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

    return m_open, m_close, max_drawdown







