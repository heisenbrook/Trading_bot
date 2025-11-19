import joblib
import plotly.graph_objects as go
import numpy as np
import torch
import os
import json
from utils.keys import get_candles, data_folder, train_data_folder_tf, fine_tuning_data_folder_tf, train_data_folder_lstm, fine_tuning_data_folder_lstm
from utils.data import BTCDataset, preprocess
from utils.plotting import plot_predictions


def make_predictions(model, data_loader, mae_close, dataset_istance, last_known_prices):
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
            

    if input_model == 'tf':
        print('Transformer model selected.')
        print('------------------------------')
        if os.path.exists(os.path.join(fine_tuning_data_folder_tf, f'td_finetuned_model.pt')):
            print('Loading fine-tuned transformer model for predictions...')
            model_path = os.path.join(fine_tuning_data_folder_tf,'td_finetuned_model.pt')
        else:
            print('Loading base transformer model for predictions...')
            model_path = os.path.join(train_data_folder_tf,'td_best_model.pt')
        model = torch.jit.load(model_path)
        model.to('cpu')
        model.eval()

        preprocessor_path = os.path.join(train_data_folder_tf, 'preprocessor.pkl')
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f'Preprocessor file not found at {preprocessor_path}')
        loaded_preprocessor = joblib.load(preprocessor_path)
        print(f'Preprocessor loaded from {preprocessor_path}')

        with open(os.path.join(train_data_folder_tf, 'best_params.json'), 'r') as f:
            best_params = json.load(f)

        if os.path.exists(os.path.join(fine_tuning_data_folder_tf, 'continual_learning_log.json')):
            with open(os.path.join(fine_tuning_data_folder_tf, 'continual_learning_log.json'), 'r') as f:
                results_dict = json.load(f)
            mae_closes = [entry['mae_close'] for entry in results_dict if entry['mae_close'] != '-']
            mae_close = mae_closes[-1]
        
        else:
            with open(os.path.join(train_data_folder_tf, 'mae_close.json'), 'r') as f:
                mae_close_dict = json.load(f)
            mae_close = mae_close_dict['mae_close']
    else:
        print('LSTM model selected.')
        print('------------------------------')
        if os.path.exists(os.path.join(fine_tuning_data_folder_lstm, f'td_finetuned_model.pt')):
            print('Loading fine-tuned LSTM model for predictions...')
            model_path = os.path.join(fine_tuning_data_folder_lstm,'td_finetuned_model.pt')
        else:
            print('Loading base LSTM model for predictions...')
            model_path = os.path.join(train_data_folder_lstm,'td_best_model.pt')
        model = torch.jit.load(model_path)
        model.to('cpu')
        model.eval()

        preprocessor_path = os.path.join(train_data_folder_lstm, 'preprocessor.pkl')
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f'Preprocessor file not found at {preprocessor_path}')
        loaded_preprocessor = joblib.load(preprocessor_path)
        print(f'Preprocessor loaded from {preprocessor_path}')

        with open(os.path.join(train_data_folder_lstm, 'best_params.json'), 'r') as f:
            best_params = json.load(f)

        if os.path.exists(os.path.join(fine_tuning_data_folder_lstm, 'continual_learning_log.json')):
            with open(os.path.join(fine_tuning_data_folder_lstm, 'continual_learning_log.json'), 'r') as f:
                results_dict = json.load(f)
            mae_closes = [entry['mae_close'] for entry in results_dict if entry['mae_close'] != '-']
            mae_close = mae_closes[-1]
            
        else:
            with open(os.path.join(train_data_folder_lstm, 'mae_close.json'), 'r') as f:
                mae_close_dict = json.load(f)
            mae_close = mae_close_dict['mae_close']

    return model, best_params, mae_close, input_model, loaded_preprocessor
            


# Main prediction routine    

btcusdt = get_candles(10000)


model, best_params, mae_close, input_model, loaded_prep = choose_model()

btcusdt = preprocess(best_params['horizon'], btcusdt, is_inference=True)
btcusdt = btcusdt.iloc[-(best_params['win_size'] + best_params['horizon']):]
btcusdt.to_csv(os.path.join(data_folder, 'btcusdt_4h_processed.csv'))

btcusdt_copy = btcusdt.copy()

processed_data = BTCDataset(btcusdt,
                           win_size=best_params['win_size'], 
                           horizon=best_params['horizon'],
                           is_training=False,
                           preprocessor=loaded_prep)

data_loader = torch.utils.data.DataLoader(processed_data)

last_known_prices = btcusdt_copy.iloc[-best_params['horizon']:][['close']].reset_index(drop=True)
pred_df = make_predictions(model, data_loader, mae_close, processed_data, last_known_prices)
if input_model.lower() == 'tf':
    plot_predictions(btcusdt_copy.iloc[-18:], pred_df)
    pred_df.to_csv(os.path.join(data_folder, 'predictions_transformer.csv'))
    print(f'Predictions saved to {os.path.join(data_folder, "predictions_transformer.csv")}')
else:
    plot_predictions(btcusdt_copy.iloc[-18:], pred_df, LSTM=True)
    pred_df.to_csv(os.path.join(data_folder, 'predictions_lstm.csv'))
    print(f'Predictions saved to {os.path.join(data_folder, "predictions_lstm.csv")}')