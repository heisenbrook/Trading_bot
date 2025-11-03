import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import torch
import os
from utils.keys import train_data_folder, fine_tuning_data_folder
from utils.plotting import plot_loss, plot_loss_fine_tuning
from datetime import datetime as dt
from tqdm import tqdm

scaler = torch.amp.GradScaler()


#============================================
# Training and evaluation functions
#============================================

def train_epoch(device, epoch, n_epochs, model, optimizer, criterion, loader):
    """
    Train the model for one epoch.
    """

    model.train()
    tot_loss = 0
    for data, label, _ in tqdm(loader,
                           desc=f'Epoch {epoch +1}/{n_epochs} | Batch Train',
                           total= len(loader),
                           leave=False,
                           ncols=80):
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()
        if device.type == 'cuda':
            with torch.autocast(device_type='cuda'):
                out = model(data)
                loss = criterion(out, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(data)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

        tot_loss += loss.item()
    
    return tot_loss/len(loader)


def eval_epoch(device, epoch, n_epochs, model, criterion, loader):
    """ 
    Evaluate the model on the validation/test set.
    """

    model.eval()
    tot_loss = 0
    with torch.no_grad():
        for data, label, _ in tqdm(loader,
                           desc=f'Epoch {epoch +1}/{n_epochs} | Batch Test',
                           total= len(loader),
                           leave=False,
                           ncols=80):
            data, label = data.to(device), label.to(device)
            out = model(data)
            loss = criterion(out, label)
            tot_loss += loss.item()
    
    return tot_loss/len(loader)


#============================================
# Main training function
#============================================

def train_test(device, n_epochs, model, optimizer, criterion, scheduler, train_loader, test_loader, fine_tuning=False):
    """         
    Main training loop with early stopping and model saving.
    """
    
    best_test_loss = float('inf')

    train_losses, test_losses = [], []
    patience = 0

    for epoch in range(n_epochs):
        train_loss = train_epoch(device, epoch, n_epochs, model, optimizer, criterion, train_loader)
        test_loss = eval_epoch(device, epoch, n_epochs, model, criterion, test_loader)
    
        scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if (epoch + 1) % (n_epochs//10) == 0 or (epoch + 1) == 1:
            x = dt.now()
            print(f'{x.strftime('%Y-%m-%d %H:%M:%S')}| Epoch {epoch + 1} | training loss:{train_loss:.5f}% | test loss:{test_loss:.5f}%')

        if test_loss < best_test_loss:
            patience = 0
            best_test_loss = test_loss
            saved_model = torch.jit.script(model)
            if fine_tuning:
                saved_model.save(os.path.join(fine_tuning_data_folder, f'td_finetuned_model.pt'))
            else:
                saved_model.save(os.path.join(train_data_folder,'td_best_model.pt'))
        elif epoch > 10 and test_loss > best_test_loss:
            patience += 1

        if epoch >30 and patience > 30:
            x = dt.now()
            print(f'{x.strftime('%Y-%m-%d %H:%M:%S')}| Epoch {epoch + 1} | training loss:{train_loss:.5f}% | test loss:{test_loss:.5f}%')
            print('Early stop')
            break
    
    if fine_tuning:
        plot_loss_fine_tuning(train_losses, test_losses)
    else:
        plot_loss(train_losses, test_losses)




