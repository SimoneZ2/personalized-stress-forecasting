"""Adaptive Stress Regressor: population LSTM and per-user online fine-tuning."""

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

FT_STEPS_P1, FT_LR_P1 = 3, 5e-4    # phase 1: output layer only
FT_STEPS_P2, FT_LR_P2 = 10, 1e-4   # phase 2: whole network


class StressLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=64, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)


def _impute(x, col_med):
    x = x.copy()
    for fi in range(x.shape[-1]):
        x[..., fi] = np.where(np.isnan(x[..., fi]), col_med[fi], x[..., fi])
    return x


def train_global(X_tr, y_tr, n_features, device, hidden_size=64, dropout=0.3, lr=1e-3,
                 max_epochs=100, batch_size=256, patience=15, huber_delta=1.0):
    """Population model; the last 15% of the training rows is used for early stopping.
    Returns the model and the per-feature training medians used for imputation."""
    model = StressLSTM(n_features, hidden_size, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.HuberLoss(delta=huber_delta)

    n_val = max(1, int(len(X_tr) * 0.15))
    X_fit, y_fit = X_tr[:-n_val].copy(), y_tr[:-n_val].copy()
    X_val, y_val = X_tr[-n_val:].copy(), y_tr[-n_val:].copy()

    col_med = np.nanmedian(X_fit.reshape(-1, X_fit.shape[-1]), axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    X_fit, X_val = _impute(X_fit, col_med), _impute(X_val, col_med)

    Xf = torch.tensor(X_fit, dtype=torch.float32).to(device)
    yf = torch.tensor(y_fit, dtype=torch.float32).to(device)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(Xf, yf), batch_size=batch_size, shuffle=False)

    best_val, best_state, wait = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        for Xb, yb in loader:
            opt.zero_grad()
            crit(model(Xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(Xv), yv).item()
        if val_loss < best_val - 1e-4:
            best_val, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, col_med, best_val


def finetune_user(model, x_seq, y_value, col_med, device, inplace=False, huber_delta=1.0):
    """Two-phase update on one (sequence, self-report) pair. With inplace=False the
    model is copied first, so the population model is never modified."""
    user_model = model if inplace else copy.deepcopy(model)
    user_model.to(device)

    X_t = torch.tensor(_impute(x_seq, col_med)[np.newaxis], dtype=torch.float32).to(device)
    y_t = torch.tensor([float(y_value)], dtype=torch.float32).to(device)
    crit = nn.HuberLoss(delta=huber_delta)

    for p in user_model.lstm.parameters():
        p.requires_grad = False
    opt1 = torch.optim.Adam(user_model.fc.parameters(), lr=FT_LR_P1, weight_decay=1e-4)
    user_model.train()
    for _ in range(FT_STEPS_P1):
        opt1.zero_grad()
        crit(user_model(X_t), y_t).backward()
        nn.utils.clip_grad_norm_(user_model.parameters(), 1.0)
        opt1.step()

    for p in user_model.lstm.parameters():
        p.requires_grad = True
    opt2 = torch.optim.Adam(user_model.parameters(), lr=FT_LR_P2, weight_decay=1e-4)
    user_model.train()
    for _ in range(FT_STEPS_P2):
        opt2.zero_grad()
        crit(user_model(X_t), y_t).backward()
        nn.utils.clip_grad_norm_(user_model.parameters(), 1.0)
        opt2.step()
    return user_model


def predict_one(model, x_seq, col_med, device, y_range=(1.0, 5.0)):
    model.eval()
    x = torch.tensor(_impute(x_seq, col_med)[np.newaxis], dtype=torch.float32).to(device)
    with torch.no_grad():
        return float(np.clip(model(x).item(), *y_range))
