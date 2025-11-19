import torch
from torch.utils.data import DataLoader, random_split
from datetime import datetime as dt
import os
from optuna import TrialPruned
from tqdm import tqdm
from utils.data import BTCDataset, preprocess
from utils.keys import train_data_folder_tf, train_data_folder_lstm, generator
from utils.train import train_epoch, eval_epoch
from utils.model import FinanceTransf, FinanceLSTM, DirectionalAccuracyLoss
from utils.plotting import plot_closes_tf, plot_closes_LSTM, plot_closes_fine_tuning_tf, plot_closes_fine_tuning_LSTM
import numpy as np
from sklearn.metrics import mean_absolute_error

#==================================================
# Hyperparameter optimization and metrics functions
#==================================================


def objective_tf(trial, device, btcusdt):
    """
    Objective function for hyperparameter optimization using Optuna.
    It defines the model architecture, training process, and evaluation metrics.
    Similar to the main training loop but adapted for hyperparameter tuning.
    """

    params = {
        'n_layers': trial.suggest_int('n_layers', 1, 4),
        'd_model': trial.suggest_int('d_model', 32, 128, step=16),
        'n_heads': trial.suggest_int('n_heads', 4, 8, step=4),
        'dim_feedforward': trial.suggest_int('dim_feedforward', 64, 512, step=64),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.1),
        'activation': trial.suggest_categorical('activation', ['relu', 'gelu']),
        'horizon': trial.suggest_int('horizon', 12, 24, step=6),
        'win_size': trial.suggest_int('win_size', 64, 256),
        'batch_size': trial.suggest_int('batch_size', 32, 128, step=16),
        'alpha': trial.suggest_float('alpha', 0.1, 0.9, step=0.1),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'n_epochs': trial.suggest_int('n_epochs', 50, 200, step=10),
        'optim': trial.suggest_categorical('optim', ['adam', 'sgd'])
    }

    btcusdt = preprocess(params['horizon'], btcusdt)
    n = len(btcusdt)
    if n < params['win_size'] + params['horizon']:
        raise TrialPruned(f'Dataset too small for the given win_size and horizon.')
    
    n_train = int(n * 0.7)
    n_test = int(n * 0.2)
    n_eval = n - n_train - n_test

    if n_train < params['win_size'] + params['horizon'] or \
    n_test < params['win_size'] + params['horizon'] or \
    n_eval < params['win_size'] + params['horizon']:
        raise TrialPruned(f'Dataset split too small for the given win_size and horizon.')
    
    train_df = btcusdt.iloc[:n_train]
    test_df = btcusdt.iloc[n_train - params['win_size']:n_train + n_test]
    eval_df = btcusdt.iloc[n_train + n_test - params['win_size']:]

    train_data = BTCDataset(train_df, 
                           win_size=params['win_size'], 
                           horizon=params['horizon'],
                           is_training=True)
    
    train_preprocessor = train_data.preprocessor

    test_data = BTCDataset(test_df, 
                          win_size=params['win_size'], 
                          horizon=params['horizon'],
                          is_training=False,
                          preprocessor=train_preprocessor)
    
    eval_data = BTCDataset(eval_df, 
                          win_size=params['win_size'], 
                          horizon=params['horizon'],
                          is_training=False,
                          preprocessor=train_preprocessor)


    train_loader = DataLoader(train_data, params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_data, params['batch_size'], shuffle=False)
    eval_loader = DataLoader(eval_data, params['batch_size'], shuffle=False)

    model = FinanceTransf(num_features= train_data.feat_cols_num,
                          n_targets=len(train_data.target_col),
                          n_layers=params['n_layers'],
                          d_model=params['d_model'],
                          n_heads=params['n_heads'],
                          dim_feedforward=params['dim_feedforward'],
                          dropout=params['dropout'],
                          activation=params['activation'],
                          horizon=params['horizon'], 
                          win_size=params['win_size'])

    model.to(device)

    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    criterion = DirectionalAccuracyLoss(params['alpha'])
    if params['optim'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-5)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9, weight_decay=1e-5)
        
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
            m_close, max_drawdown = optim_testing(device, model, eval_loader, eval_data, epoch, params['n_epochs'])
            tot_loss = m_close + max_drawdown
            trial.report(tot_loss, epoch)
            if trial.should_prune():
                raise TrialPruned(f'Epoch {epoch + 1} | training loss:{train_loss:.5f}% | test loss:{test_loss:.5f}% | MAE Close: ${m_close:.2f} | Max Drawdown: ${max_drawdown:.2f}')

        if epoch >30 and patience > 30:
            x = dt.now()
            print(f'{x.strftime('%Y-%m-%d %H:%M:%S')}| Epoch {epoch + 1} | training loss:{train_loss:.5f}% | test loss:{test_loss:.5f}%')
            print('Early stop')
            break

    m_close, max_drawdown = optim_testing(device, model, eval_loader, eval_data, epoch, params['n_epochs'])
    loss = m_close + max_drawdown
    print(f'max drawdown %: {max_drawdown * 100:.2f}%')
    print(f'MAE Close %: {m_close * 100:.2f}%')
    
    return loss

def objective_lstm(trial, device, btcusdt):
    """
    Objective function for hyperparameter optimization using Optuna.
    It defines the model architecture, training process, and evaluation metrics.
    Similar to the main training loop but adapted for hyperparameter tuning.
    Adapted for LSTM model.
    """

    params = {
        'hidden_size': trial.suggest_int('hidden_size', 32, 128, step=16),
        'num_layers': trial.suggest_int('n_layers', 2, 6, step=1),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.1),
        'horizon': trial.suggest_int('horizon', 12, 24, step=6),
        'win_size': trial.suggest_int('win_size', 64, 256, step=16),
        'batch_size': trial.suggest_int('batch_size', 32, 128, step=16),
        'alpha': trial.suggest_float('alpha', 0.1, 0.9, step=0.1),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'n_epochs': trial.suggest_int('n_epochs', 50, 200, step=10),
        'optim': trial.suggest_categorical('optim', ['adam', 'sgd'])
    }

    btcusdt = preprocess(params['horizon'], btcusdt)
    n = len(btcusdt)
    if n < params['win_size'] + params['horizon']:
        raise TrialPruned(f'Dataset too small for the given win_size and horizon.')
    
    n_train = int(n * 0.7)
    n_test = int(n * 0.2)
    n_eval = n - n_train - n_test

    if n_train < params['win_size'] + params['horizon'] or \
    n_test < params['win_size'] + params['horizon'] or \
    n_eval < params['win_size'] + params['horizon']:
        raise TrialPruned(f'Dataset split too small for the given win_size and horizon.')
    
    train_df = btcusdt.iloc[:n_train]
    test_df = btcusdt.iloc[n_train - params['win_size']:n_train + n_test]
    eval_df = btcusdt.iloc[n_train + n_test - params['win_size']:]

    train_data = BTCDataset(train_df, 
                           win_size=params['win_size'], 
                           horizon=params['horizon'],
                           is_training=True)
    
    train_preprocessor = train_data.preprocessor

    test_data = BTCDataset(test_df, 
                          win_size=params['win_size'], 
                          horizon=params['horizon'],
                          is_training=False,
                          preprocessor=train_preprocessor)
    
    eval_data = BTCDataset(eval_df, 
                          win_size=params['win_size'], 
                          horizon=params['horizon'],
                          is_training=False,
                          preprocessor=train_preprocessor)


    train_loader = DataLoader(train_data, params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_data, params['batch_size'], shuffle=False)
    eval_loader = DataLoader(eval_data, params['batch_size'], shuffle=False)

    model = FinanceLSTM(input_size=train_data.feat_cols_tot,
                        hidden_size=params['hidden_size'],
                        num_layers=params['num_layers'],
                        n_targets=len(train_data.target_col),
                        dropout=params['dropout'],
                        horizon=params['horizon'])

    model.to(device)

    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    criterion = DirectionalAccuracyLoss(params['alpha'])
    if params['optim'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-5)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9, weight_decay=1e-5)
        
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
            m_close, max_drawdown = optim_testing(device, model, eval_loader, eval_data, epoch, params['n_epochs'])
            tot_loss = m_close + max_drawdown
            trial.report(tot_loss, epoch)
            if trial.should_prune():
                raise TrialPruned(f'Epoch {epoch + 1} | training loss:{train_loss:.5f}% | test loss:{test_loss:.5f}% | MAE Close: ${m_close:.2f} | Max Drawdown: ${max_drawdown:.2f}')

        if epoch >30 and patience > 30:
            x = dt.now()
            print(f'{x.strftime('%Y-%m-%d %H:%M:%S')}| Epoch {epoch + 1} | training loss:{train_loss:.5f}% | test loss:{test_loss:.5f}%')
            print('Early stop')
            break

    m_close, max_drawdown = optim_testing(device, model, eval_loader, eval_data, epoch, params['n_epochs'])
    loss = m_close + max_drawdown
    print(f'max drawdown %: {max_drawdown * 100:.2f}%')
    print(f'MAE Close %: {m_close * 100:.2f}%')
    
    return loss


#============================================
# Testing functions
#============================================   

def metrics(targets, preds):
    """
    Calculate evaluation metrics: Mean Absolute Error for open and close prices, and maximum drawdown.
    """

    mae_close = mean_absolute_error(targets['next_close'], preds['next_close'])
    max_drawdown = np.max(np.abs(targets['next_close'] - preds['next_close']))

    return mae_close, max_drawdown


def testing(device, model, loader, full_data, fine_tuning=False, lstm=False):
    """
    Evaluate the model on the evaluation dataset and return evaluation metrics.       
    """

    model.eval()
    all_preds, all_targets, all_time = [], [], []

    with torch.no_grad():
        for data, label, time in tqdm(loader,
                           desc=f'Evaluating model',
                           total= len(loader),
                           leave=False,
                           ncols=80):
            data = data.to(device)
            preds = model(data) 
            all_preds.append(preds.cpu().numpy())
            all_targets.append(label.cpu().numpy())
            all_time.append(time)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_time = np.concatenate(all_time, axis=0)

    all_preds = all_preds.reshape(-1)
    all_targets = all_targets.reshape(-1)
    all_time = all_time.reshape(-1)

    preds_real = full_data.denorm_pred(all_preds, all_time)
    targets_real = full_data.denorm_pred(all_targets, all_time)

    if lstm:
        preds_real.sort_index().to_csv(os.path.join(train_data_folder_lstm,'preds_real.csv'))
        targets_real.sort_index().to_csv(os.path.join(train_data_folder_lstm,'targets_real.csv'))
    else:
        preds_real.sort_index().to_csv(os.path.join(train_data_folder_tf,'preds_real.csv'))
        targets_real.sort_index().to_csv(os.path.join(train_data_folder_tf,'targets_real.csv'))

    if fine_tuning:
        if lstm:
            plot_closes_fine_tuning_LSTM(targets_real, preds_real)
        else:
            plot_closes_fine_tuning_tf(targets_real, preds_real)
    elif lstm:
        plot_closes_LSTM(targets_real, preds_real)
    else:
        plot_closes_tf(targets_real, preds_real)

    m_close, max_drawdown = metrics(targets_real, preds_real)

    return m_close, max_drawdown


def optim_testing(device, model, loader, full_data, epoch, n_epochs):
    """
    Evaluate the model during hyperparameter optimization and return evaluation metrics.       
    """
    
    model.eval()
    all_preds, all_targets, all_time = [], [], []

    with torch.no_grad():
        for data, label, time in tqdm(loader,
                           desc=f'Epoch {epoch +1}/{n_epochs} | Eval Test',
                           total= len(loader),
                           leave=False,
                           ncols=80):
            data = data.to(device)
            preds = model(data) 
            all_preds.append(preds.cpu().numpy())
            all_targets.append(label.cpu().numpy())
            all_time.append(time)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_time = np.concatenate(all_time, axis=0)

    all_preds = all_preds.reshape(-1)
    all_targets = all_targets.reshape(-1)
    all_time = all_time.reshape(-1)

    preds_real = full_data.denorm_pred(all_preds, all_time)
    targets_real = full_data.denorm_pred(all_targets, all_time)

    m_close, max_drawdown = metrics(targets_real, preds_real)

    return m_close, max_drawdown







