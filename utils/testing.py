import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

def metrics(targets, preds):

    mae_open = mean_absolute_error(targets[:,0], preds[:,0])
    mae_close = mean_absolute_error(targets[:,1], preds[:,1])

    return mae_open, mae_close

def plot_one(targets, preds):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(targets[1000:2000, 0], label='True Open')
    plt.plot(preds[1000:2000, 0], label='Pred Open')
    plt.title('Open Prices')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(targets[1000:2000, 1], label='True Close')
    plt.plot(preds[1000:2000, 1], label='Pred Close')
    plt.title('Close Prices')
    plt.legend()

    plt.tight_layout()
    plt.savefig('Pred_vs_real.png')
    plt.close()


def testing(device, model, loader, full_data):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data, label in loader:
            data = data.to(device)
            preds = model(data)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(label.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    all_preds = all_preds.reshape(-1,2)
    all_targets = all_targets.reshape(-1,2)

    preds_real = full_data.denorm_pred(all_preds)
    targets_real = full_data.denorm_pred(all_targets)

    plot_one(targets_real, preds_real)

    m_open, m_close = metrics(targets_real, preds_real)

    print(f'MAE Open: ${m_open:.2f}')
    print(f'MAE Close: ${m_close:.2f}')






