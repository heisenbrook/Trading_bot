from tvDatafeed import TvDatafeed, Interval
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from model import FinanceTransf
from keys import user, psw
from data import BTCDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tv = TvDatafeed(user, psw)

btcusdt = tv.get_hist(symbol='BTCUSDT', 
                      exchange='BINANCE', 
                      interval=Interval.in_4_hour, 
                      n_bars=10000,
                      extended_session=True)

full_data = BTCDataset(btcusdt)


train_data, test_data = random_split(full_data, [0.8 , 0.2])

batch_size = 64
train_loader = DataLoader(train_data, batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size, shuffle=False)
print(len(train_loader))
print(len(test_loader))

td_bot = FinanceTransf(
    num_features=len(full_data.feat_cols),
    n_targets=len(full_data.target_col),
    n_layers=2
)
td_bot.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(td_bot.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

def train_epoch(model, loader):
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

def eval_epoch(model, loader):
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

# main training loop

n_epochs = 50
best_test_loss = float('inf')

train_losses, test_losses = [], []

for epoch in range(n_epochs):
    train_loss = train_epoch(td_bot, train_loader)
    test_loss = train_epoch(td_bot, test_loader)
    
    scheduler.step(test_loss)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f'Epoch {epoch + 1} | training loss:{train_loss:.5f}% | test loss:{test_loss:.5f}%')

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(td_bot.state_dict(), "td_best_model.pth")
    elif epoch >10 and test_loss > best_test_loss * 1.1:
        print('Early stop')
        break

plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.title("Training History")
plt.show()

