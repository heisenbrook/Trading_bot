import joblib
import plotly.graph_objects as go
import numpy as np
import torch
import os
import json
import pandas as pd
from utils.keys import get_candles, data_folder, classification_data_folder, train_data_folder_tf, fine_tuning_data_folder_tf, classification_train_data_folder_tf, classification_train_data_folder_lstm, classification_fine_tuning_data_folder_lstm, classification_fine_tuning_data_folder_tf, train_data_folder_lstm, fine_tuning_data_folder_lstm
from utils.data import BTCDataset, preprocess
from utils.plotting import plot_predictions, plot_class_signals

def choose_folder(input_model, input_class):
    """
    Chooses the correct folder paths based on model and task type.
    """
    if input_model == 'tf' and input_class == 'regr':
        train_data_folder = train_data_folder_tf
        fine_tuning_data_folder = fine_tuning_data_folder_tf
    elif input_model == 'tf' and input_class == 'class':
        train_data_folder = classification_train_data_folder_tf
        fine_tuning_data_folder = classification_fine_tuning_data_folder_tf
    elif input_model == 'lstm' and input_class == 'regr':
        train_data_folder = train_data_folder_lstm
        fine_tuning_data_folder = fine_tuning_data_folder_lstm
    else:
        train_data_folder = classification_train_data_folder_lstm
        fine_tuning_data_folder = classification_fine_tuning_data_folder_lstm

    return train_data_folder, fine_tuning_data_folder


def make_predictions(model, data_loader, mae_close: float|None, dataset_istance, last_known_prices, input_class):
    """
    Makes predictions using the trained model and returns a DataFrame with denormalized predictions.
    """
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

    df = dataset_istance.denorm_pred(all_preds, all_time)

    if input_class == 'class':
        probs = torch.sigmoid(torch.tensor(all_preds)).numpy()
        df['prob_up'] = probs

        threshold_buy = 0.6
        threshold_sell = 0.4
        df['signal'] = 'HOLD'
        df.loc[df['prob_up'] > threshold_buy, 'signal'] = 'BUY'
        df.loc[df['prob_up'] < threshold_sell, 'signal'] = 'SELL'
        df.dropna(inplace=True)
    else:
        current_prices = last_known_prices['close'].values
        pred_returns = df['next_close'].values 
        pred_prices = current_prices * np.exp(pred_returns)
        df['next_close'] = pred_prices
        df['next_open'] = df['next_close'].shift(1)
        df['range_low'] = df['next_close'] * np.exp(-mae_close)
        df['range_high'] = df['next_close'] * np.exp(mae_close)
        df = df.rename(columns={'next_open':'open','next_close':'close'})
        df = df.loc[:, ['range_low','open','close','range_high']]
        df.dropna(inplace=True)

    return df

def choose_model():
    """
    Chooses architecture based on saved model files and input from user.
    """
    valid_input = False
    while valid_input == False:
        try:
            input_model = input('Select model to use for predictions (tf/lstm): ').strip().lower()
            if input_model not in ['tf', 'lstm']:
                raise ValueError('Invalid model type. Choose "tf" or "lstm".')
            valid_input = True
        except ValueError as e:
            print(e)
    
    valid_input = False
    while valid_input == False:
        try:
            input_class = input('Select between classification or regression (class/regr): ').strip().lower()
            if input_class not in ['class', 'regr']:
                raise ValueError('Invalid model type. Choose "class" or "regr".')
            valid_input = True
        except ValueError as e:
            print(e)

    train_data_folder, fine_tuning_data_folder = choose_folder(input_model, input_class)        

    print('------------------------------')
    if input_model == 'tf':
        print('Transformer model selected.')
        model_type = 'Transformer'
    else:
        print('LSTM model selected.')
        model_type = 'LSTM'

    if input_class == 'class':
        print('classification task selected.')
    else:
        print('regression task selected.')
    print('------------------------------')

    if os.path.exists(os.path.join(fine_tuning_data_folder, f'td_finetuned_model.pt')):
        print(f'Loading fine-tuned {model_type} model for predictions...')
        model_path = os.path.join(fine_tuning_data_folder,'td_finetuned_model.pt')

    else:
        print(f'Loading base {model_type} model for predictions...')
        model_path = os.path.join(train_data_folder,'td_best_model.pt')

    model = torch.jit.load(model_path)
    model.to('cpu')
    model.eval()

    preprocessor_path = os.path.join(train_data_folder, 'preprocessor.pkl')
    if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f'Preprocessor file not found at {preprocessor_path}')
    loaded_preprocessor = joblib.load(preprocessor_path)
    print(f'Preprocessor loaded from {preprocessor_path}')

    with open(os.path.join(train_data_folder, 'best_params.json'), 'r') as f:
        best_params = json.load(f)

    if os.path.exists(os.path.join(fine_tuning_data_folder, 'continual_learning_log.json')):
        with open(os.path.join(fine_tuning_data_folder, 'continual_learning_log.json'), 'r') as f:
            results_dict = json.load(f)
        if input_class == 'regr':
            mae_closes = [entry['mae_close'] for entry in results_dict if entry['mae_close'] != '-']
            mae_close = mae_closes[-1]
        else:
            mae_close = None
    else:
        with open(os.path.join(train_data_folder, 'metrics.json'), 'r') as f:
            metrics_dict = json.load(f)
        if input_class == 'regr':
            mae_close = metrics_dict['mae_close']
        else:
            mae_close = None

    return model, best_params, mae_close, input_model, input_class, loaded_preprocessor
            


# Main prediction routine    

btcusdt = get_candles(10000)


model, best_params, mae_close, input_model, input_class, loaded_prep = choose_model()

btcusdt_copy = btcusdt.copy()
if input_class == 'class':

    btcusdt = preprocess(best_params['horizon'], btcusdt, is_inference=True, is_classification=True)
    btcusdt = btcusdt.iloc[-(best_params['win_size'] + best_params['horizon']):]
    btcusdt.to_csv(os.path.join(data_folder, f'btcusdt_4h_processed_{input_class}.csv'))

    processed_data = BTCDataset(btcusdt,
                           win_size=best_params['win_size'], 
                           horizon=best_params['horizon'],
                           is_training=False,
                           is_classification=True,
                           preprocessor=loaded_prep)
else:

    btcusdt = preprocess(best_params['horizon'], btcusdt, is_inference=True)
    btcusdt = btcusdt.iloc[-(best_params['win_size'] + best_params['horizon']):]
    btcusdt.to_csv(os.path.join(data_folder, f'btcusdt_4h_processed_{input_class}.csv'))

    processed_data = BTCDataset(btcusdt,
                            win_size=best_params['win_size'], 
                            horizon=best_params['horizon'],
                            is_training=False,
                            preprocessor=loaded_prep)

data_loader = torch.utils.data.DataLoader(processed_data)

last_known_prices = btcusdt_copy.iloc[best_params['win_size']-1:-best_params['horizon']][['close']].reset_index(drop=True)
pred_df = make_predictions(model, data_loader, mae_close, processed_data, last_known_prices, input_class)
if input_model == 'lstm':
    name = 'LSTM'
else:
    name = 'Transformer'

if input_class.lower() == 'regr':
    last_real_row = btcusdt_copy.iloc[-1]
    last_real_date = btcusdt_copy.index[-1]

    bridge_row = pd.DataFrame({
        'range_low': last_real_row['low'],
        'open': last_real_row['open'],
        'close': last_real_row['close'],
        'range_high': last_real_row['high']
    }, index=[last_real_date])

    pred_df = pd.concat([bridge_row, pred_df])

    plot_predictions(btcusdt_copy.iloc[-18:], pred_df, data_folder, input_model)
    pred_df.to_csv(os.path.join(data_folder, f'predictions_regression_{name}.csv'))
    print(f'Predictions saved to {os.path.join(data_folder, f'predictions_regression_{name}.csv')}')

else:
    plot_class_signals(btcusdt_copy.iloc[-100:], pred_df, classification_data_folder, input_model)
    pred_df.to_csv(os.path.join(classification_data_folder, f'predictions_classification_{name}.csv'))
    print(f'Predictions saved to {os.path.join(classification_data_folder, f'predictions_classification_{name}.csv')}')
