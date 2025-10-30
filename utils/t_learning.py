from utils.keys import train_data_folder
import numpy as np
import torch
import torch.nn as nn
import os
from utils.data import BTCDataset
from utils.testing import plot_closes, metrics


class Transfer_learner(nn.Module):
    """
    Transfer Learning wrapper for the FinanceTransf model.
    Freezes all layers except the final output layer for fine-tuning on new data.
    """
    def __init__(self, model_path, freeze_base=True):
        super().__init__()
        self.model_path = model_path
        self.freeze_base = freeze_base
        self.base_model = None
    
    def load_freeze_model(self):
        print('Loading and freezing base model...')
        total_params = 0
        trainaìble_params = 0

        self.base_model = torch.jit.load(self.model_path)
        if self.freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
                total_params += param.numel()
            for param in self.base_model.out.parameters():
                param.requires_grad = True
                trainaìble_params += param.numel()
            
        frozen_params = total_params - trainaìble_params
        print(f'Total parameters: {total_params}')
        print(f'Frozen parameters: {frozen_params}')
        print(f'Trainable parameters: {trainaìble_params}')
                
        self.base_model.train()
        return self.base_model

    def forward(self, x):
        self.base_model = self.load_freeze_model()
        return self.base_model(x)
    

class Continous_learning(nn.Module):
    """
    Continual Learning wrapper for the FinanceTransf model.
    Allows training on new tasks while retaining knowledge from previous tasks.
    """
    def __init__(self, model_path, data_loader, retrain_interval=7):
        super().__init__()
        self.model_path = model_path
        self.data_loader = data_loader
        self.retrain_interval = retrain_interval
        self.retrain_h = []
    
    def should_retrain(self, current_h):
        if current_h % self.retrain_interval == 0 and current_h not in self.retrain_h:
            self.retrain_h.append(current_h)
            return True
        else:
            return False
    