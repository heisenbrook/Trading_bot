import json
import os
import torch
import optuna
from optuna.pruners import MedianPruner
from utils.keys import tv, train_data_folder_tf, train_data_folder_lstm, classification_train_data_folder_lstm, classification_train_data_folder_tf, get_candles
from utils.testing import objective

#============================================
# Optimization script
#============================================

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
btcusdt  = get_candles(10000)

# Set up Optuna study
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=15, interval_steps=5)
sampler = optuna.samplers.TPESampler(seed=42)

valid_input = False
while valid_input == False:
    try:
        input_model = input('Select model to perform parameters optimization (tf/lstm): ').strip().lower()
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

if input_class == 'class':
    study = optuna.create_study(direction='maximize', 
                                sampler=sampler, 
                                pruner=pruner,
                                study_name='BTC_Transf_Classification',
                                load_if_exists=True)
else:
    study = optuna.create_study(direction='minimize', 
                                sampler=sampler, 
                                pruner=pruner, 
                                study_name='BTC_Transf_regression',
                                load_if_exists=True)

# Run optimization
if input_model == 'tf' and input_class == 'regr':
    study.optimize(lambda trial: objective(trial, device, btcusdt), n_trials=200)
    folder = train_data_folder_tf
elif input_model == 'tf' and input_class == 'class':
    study.optimize(lambda trial: objective(trial, device, btcusdt, is_classification=True), n_trials=200)
    folder = classification_train_data_folder_tf
elif input_model == 'lstm' and input_class == 'regr':
    study.optimize(lambda trial: objective(trial, device, btcusdt, lstm=True), n_trials=200)
    folder = train_data_folder_lstm
else:
    study.optimize(lambda trial: objective(trial, device, btcusdt, lstm=True, is_classification=True), n_trials=200)
    folder = classification_train_data_folder_lstm

best_params = study.best_params
best_value = study.best_value

print(f'Best trial: {best_value}')
print(f'Best params: {best_params}')

# Save the best parameters to a JSON file
with open(os.path.join(folder, 'best_params.json'), 'w') as f:
    json.dump(best_params, f, indent=4)






