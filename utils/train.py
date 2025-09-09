import plotly.express as px
import pandas as pd
import torch
import os
from utils.keys import data_folder
from datetime import datetime as dt
from tqdm import tqdm

scaler = torch.amp.GradScaler()

def plot_loss(train_losses, test_losses):
    df = pd.DataFrame(dict(train_loss=train_losses, test_loss=test_losses))
    fig = px.line(df, labels={'index': 'Epochs', 'value': 'Loss'},
                  title='Training and Test Loss Over Epochs')
    fig.update_layout(xaxis_title='Epochs', yaxis_title='Loss') 
    fig.write_image(os.path.join(data_folder, 'Training_loss.png'))

def train_epoch(device, epoch, n_epochs, model, optimizer, criterion, loader):
    model.train()
    tot_loss = 0
    for data, label, _ in tqdm(loader,
                           desc=f'Epoch {epoch +1}/{n_epochs} | Batch Train',
                           total= len(loader),
                           leave=False,
                           ncols=80):
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda'):
            out = model(data)
            loss = criterion(out, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        tot_loss += loss.item()
    
    return tot_loss/len(loader)

def train_epoch_test(device, epoch, n_epochs, model, optimizer, criterion, loader):
    model.train()
    tot_loss = 0
    for data, label, _ in tqdm(loader,
                           desc=f'Epoch {epoch +1}/{n_epochs} | Batch Train',
                           total= len(loader),
                           leave=False,
                           ncols=80):
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda'):
            out = model(data)
            loss = criterion(out, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        tot_loss += loss.item()
    
    return tot_loss/len(loader)

def eval_epoch(device, epoch, n_epochs, model, criterion, loader):
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

def eval_epoch_test(device, epoch, n_epochs, model, criterion, loader):
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

# main training function

def train_test(device, n_epochs, model, optimizer, criterion, scheduler, train_loader, test_loader):
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
            saved_model.save(os.path.join(data_folder,'td_best_model.pt'))
        elif epoch > 10 and test_loss > best_test_loss:
            patience += 1

        if epoch >30 and patience > 30:
            x = dt.now()
            print(f'{x.strftime('%Y-%m-%d %H:%M:%S')}| Epoch {epoch + 1} | training loss:{train_loss:.5f}% | test loss:{test_loss:.5f}%')
            print('Early stop')
            break
    
    plot_loss(train_losses, test_losses)




