from tvDatafeed import TvDatafeed, Interval
from datetime import datetime as dt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
import json
import schedule
from utils.keys import user, psw, train_data_folder, fine_tuning_data_folder, generator
from utils.data import BTCDataset, preprocess
from utils.train import train_test
from utils.testing import testing
from utils.model import DirectionalAccuracyLoss

#===========================================================
# Transfer Learning modules and continual scheduled learning
#===========================================================

with open(os.path.join(train_data_folder, 'best_params.json'), 'r') as f:
    best_params = json.load(f)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class Transfer_learner(nn.Module):
    """
    Transfer Learning wrapper for the FinanceTransf model.
    Freezes all layers except the final output layer for fine-tuning on new data.
    """
    def __init__(self, model_path, train_loader, test_loader, eval_loader, freeze_base=True):
        super().__init__()
        self.model_path = model_path
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.eval_loader = eval_loader
        self.freeze_base = freeze_base
        self.base_model = None
    
    def load_freeze_model(self):
        print('Loading and freezing base model...')
        total_params = 0
        trainable_params = 0

        self.base_model = torch.jit.load(self.model_path, map_location=device) 
        self.base_model.to(device)       
        self.base_model.train()
        
        if self.freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
                total_params += param.numel()
            try:
                for param in self.base_model.out.parameters():
                    param.requires_grad = True
                    trainable_params += param.numel()
            except AttributeError:
                print("Warning: 'out' layer not found. Unfreezing all parameters.")
                for param in self.base_model.parameters():
                    param.requires_grad = True
                    trainable_params += param.numel()
            
        frozen_params = total_params - trainable_params
        print(f'Total parameters: {total_params}')
        print(f'Frozen parameters: {frozen_params}')
        print(f'Trainable parameters: {trainable_params}')

        return self.base_model
    
    def fine_tune(self, n_epochs, optimizer, criterion, scheduler):
        print('Starting fine-tuning...')
        train_test(device, 
                   n_epochs, 
                   self.base_model, 
                   optimizer, 
                   criterion, 
                   scheduler, 
                   self.train_loader, 
                   self.test_loader, 
                   fine_tuning=True)


class Continous_learning(nn.Module):
    """
    Continual Learning wrapper for the FinanceTransf model.
    Allows training on new tasks while retaining knowledge from previous tasks.
    """
    def __init__(self, model_path, current_date, retrain_interval=7):
        super().__init__()
        self.model_path = model_path
        self.retrain_interval = retrain_interval
        self.n_candles = retrain_interval * 6  # Assuming 4-hour candles
        self.retrain_h = []
        self.current_date = current_date
        self.tv = TvDatafeed(user, psw)

        self.btcusdt = self.tv.get_hist(symbol='BTCUSDT', 
                      exchange='BINANCE', 
                      interval=Interval.in_4_hour, 
                      n_bars= self.n_candles * 10,
                      extended_session=True)
        
        self.btcusdt = preprocess(best_params['horizon'], self.btcusdt)
        self.data = BTCDataset(self.btcusdt,
                           win_size=best_params['win_size'], 
                           horizon=best_params['horizon'],
                           is_training=True)

    def should_retrain(self):
        if not self.retrain_h:
            return True
        
        last_retrain_h = max(self.retrain_h)
        days_since_last_retrain = (self.current_date - last_retrain_h).days

        return days_since_last_retrain >= self.retrain_interval

    def get_data(self): 
        train_data, test_data, eval_data = random_split(self.data, [0.7 , 0.2, 0.1], generator=generator)
        
        train_loader = DataLoader(train_data, best_params['batch_size'], shuffle=False, pin_memory=False, persistent_workers=False)
        test_loader = DataLoader(test_data, best_params['batch_size'], shuffle=False, pin_memory=False, persistent_workers=False)
        eval_loader = DataLoader(eval_data, best_params['batch_size'], shuffle=False, pin_memory=False, persistent_workers=False)

        return train_loader, test_loader, eval_loader

    def continous_learning(self):
        if not self.should_retrain():
            print(f'Last retraining {max(self.retrain_h)}. No retraining needed as of {self.current_date}.')
            return None
        
        print(f'Retraining model as of {self.current_date}...')

        train_loader, test_loader, eval_loader = self.get_data()
        
        fine_tuner = Transfer_learner(self.model_path, train_loader, test_loader, eval_loader, freeze_base=True)
        model = fine_tuner.load_freeze_model()

        criterion = DirectionalAccuracyLoss(best_params['alpha'])
        if best_params['optim'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=1e-5)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=best_params['lr'], momentum=0.9, weight_decay=1e-5)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, mode='min', factor=0.5)

        fine_tuner.fine_tune(best_params['n_epochs'], optimizer, criterion, scheduler)
        print('Fine-tuning completed.')
        print(f'Saving fine-tuned model in {fine_tuning_data_folder}')
        print('Saving current retraining date...')
        self.retrain_h.append(self.current_date)
        print('Evaluating fine-tuned model on evaluation set...')
        mae_close, max_drawdown = testing(device, model, eval_loader, self.data, fine_tuning=True)
        print(f'MAE Close after fine-tuning: ${mae_close:.2f}')
        print(f'Max Drawdown after fine-tuning: ${max_drawdown:.2f}')

        return mae_close, max_drawdown

def daily_learning_routine():
    """
    Routine to perform continual learning on a daily basis.
    """
    logger_file = {}
    if os.path.exists(os.path.join(fine_tuning_data_folder, f'td_finetuned_model.pt')):
        model_path = os.path.join(fine_tuning_data_folder,'td_finetuned_model.pt')
    else:
        model_path = os.path.join(train_data_folder,'td_best_model.pt')
    current_date = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    continual_learner = Continous_learning(model_path, current_date)
    if continual_learner.should_retrain():
        print(f'Performing continual learning for date: {current_date}')
        mae_close, max_drawdown = continual_learner.continous_learning()
        result = {
            'date': current_date,
            'retraining_done': True,
            'mae_close': float(mae_close),
            'max_drawdown': float(max_drawdown)
        }
    else:
        print(f'No retraining needed for date: {current_date}')
        result = {
            'date': current_date,
            'retraining_done': False,
            'mae_close': '-',
            'max_drawdown': '-'
        }

    logger_file.update(result)
    
    with open(os.path.join(fine_tuning_data_folder, f'results_fine_tuning.json'), 'w') as f:
        json.dump(logger_file, f, indent=4)

daily_learning_routine()
#schedule.every().day.at('10:40').do(daily_learning_routine)


    