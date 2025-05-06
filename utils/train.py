import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

def train_epoch(device, model, optimizer, criterion, loader):
    model.train()
    tot_loss = 0
    for data, label in tqdm(loader,
                           desc='Batch Train',
                           total= len(loader),
                           leave=False,
                           ncols=80):
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()
    
    return tot_loss/len(loader)

def eval_epoch(device, model, criterion, loader):
    model.eval()
    tot_loss = 0
    with torch.no_grad():
        for data, label in tqdm(loader,
                           desc='Batch Eval',
                           total= len(loader),
                           leave=False,
                           ncols=80):
            data, label = data.to(device), label.to(device)
            out = model(data)
            tot_loss += criterion(out, label).item()
    
    return tot_loss/len(loader)

# main training function

def train_test(device, model, optimizer, criterion, scheduler, train_loader, test_loader):
    n_epochs = 50
    best_test_loss = float('inf')

    train_losses, test_losses = [], []

    for epoch in range(n_epochs):
        train_loss = train_epoch(device, model, optimizer, criterion, train_loader)
        test_loss = eval_epoch(device, model, criterion, test_loader)
    
        scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f'Epoch {epoch + 1} | training loss:{train_loss:.5f}% | test loss:{test_loss:.5f}%')

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'td_best_model.pth')
        elif epoch >10 and test_loss > best_test_loss * 1.1:
            print('Early stop')
            break

    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.title('Training History')
    plt.show()



