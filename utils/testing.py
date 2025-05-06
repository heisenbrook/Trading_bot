import torch
import numpy as np
from sklearn.metrics import mean_absolute_error
from model import FinanceTransf
from train import full_data, eval_data
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def metrics(model, loader):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data, label in loader:
            data = data.to(device)
            preds = model(data)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(label)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    preds_real = full_data.denorm_pred(all_preds.reshape(-1,2))
    targets_real = full_data.denorm_pred(all_targets.reshape(-1,2))

    mae_open = mean_absolute_error(targets_real[:,0], preds_real[:,0])
    mae_close = mean_absolute_error(targets_real[:,1], preds_real[:,1])

    return mae_open, mae_close

def testing_and_saving(eval_data):
    td_bot = FinanceTransf(
        num_features=len(full_data.feat_cols),
        n_targets=len(full_data.target_col),
        n_layers=4
    )  

    td_bot.load_state_dict(torch.load('td_best_model.pth'))
    td_bot.eval()

    batch = 64
    eval_loader = DataLoader(eval_data, batch_size=batch)

    x_eval, y_eval = next(iter(eval_loader))
    x_eval = x_eval.to(device)

    with torch.no_grad():
        y_pred = td_bot(x_eval)

    x_eval = x_eval.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

