import matplotlib.pyplot as plt
import torch
from datetime import datetime as dt
from tqdm import tqdm

scaler = torch.amp.GradScaler()

def train_epoch(device, epoch, model, optimizer, criterion, loader):
    model.train()
    tot_loss = 0
    for data, label in tqdm(loader,
                           desc=f'Epoch {epoch +1} | Batch Train',
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

def eval_epoch(device, epoch, model, criterion, loader):
    model.eval()
    tot_loss = 0
    with torch.no_grad():
        for data, label in tqdm(loader,
                           desc=f'Epoch {epoch +1} | Batch Test',
                           total= len(loader),
                           leave=False,
                           ncols=80):
            data, label = data.to(device), label.to(device)
            out = model(data)
            tot_loss += criterion(out, label).item()
    
    return tot_loss/len(loader)

# main training function

def train_test(device, n_epochs, model, optimizer, criterion, scheduler, train_loader, test_loader):
    best_test_loss = float('inf')

    train_losses, test_losses = [], []

    for epoch in range(n_epochs):
        train_loss = train_epoch(device, epoch, model, optimizer, criterion, train_loader)
        test_loss = eval_epoch(device, epoch, model, criterion, test_loader)
    
        scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if (epoch + 1) % (n_epochs//10) == 0 or (epoch + 1) == 1:
            x = dt.now()
            print(f'{x.strftime('%Y-%m-%d %H:%M:%S')}| Epoch {epoch + 1} | training loss:{train_loss:.5f}% | test loss:{test_loss:.5f}%')

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'td_best_model.pth')
        elif epoch >10 and test_loss > best_test_loss * 1.05:
            print(f'{x.strftime('%Y-%m-%d %H:%M:%S')}| Epoch {epoch + 1} | training loss:{train_loss:.5f}% | test loss:{test_loss:.5f}%')
            print('Early stop')
            break

    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.title('Training History')
    plt.savefig('Training_loss.png')
    plt.close()



