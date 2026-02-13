# lstm_baseline.py
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMForecaster(nn.Module):
    """Vanilla LSTM for comparison with TEMPO."""

    def __init__(self, input_size, hidden_size=128, num_layers=2, horizon=96):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions


def train_lstm(model, X_train, y_train, epochs=20, batch_size=64, lr=1e-3):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)

    dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        avg_loss = total_loss / len(dataset)
        print(f"  Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.6f}")

    return model


def evaluate_lstm(model, X_test, y_test, batch_size=256):
    device = next(model.parameters()).device
    model.eval()

    dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    loader = DataLoader(dataset, batch_size=batch_size)

    preds, targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).cpu().numpy()
            preds.append(pred)
            targets.append(y_batch.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mae = np.mean(np.abs(preds - targets))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}


def main():
    data_dir = Path(__file__).resolve().parent.parent / 'data' / 'processed'

    print("Loading processed data...")
    X_train = np.load(data_dir / 'train_X.npy')
    y_train = np.load(data_dir / 'train_y.npy')

    test_sites = ['UK-AMo', 'SE-Htm']
    test_data = {}
    for site in test_sites:
        test_data[site] = {
            'X': np.load(data_dir / f'test_{site}_X.npy'),
            'y': np.load(data_dir / f'test_{site}_y.npy'),
        }

    input_size = X_train.shape[2]  # number of features
    print(f"Training data: {X_train.shape[0]} samples, "
          f"{X_train.shape[1]} timesteps, {input_size} features")

    model = LSTMForecaster(input_size=input_size, horizon=y_train.shape[1])

    print("\nTraining LSTM...")
    model = train_lstm(model, X_train, y_train)

    print("\n" + "=" * 60)
    print("LSTM BASELINE RESULTS")
    print("=" * 60)

    for site in test_sites:
        metrics = evaluate_lstm(model, test_data[site]['X'], test_data[site]['y'])
        n = len(test_data[site]['X'])
        print(f"\n--- {site} ({n} samples) ---")
        print(f"  LSTM:  RMSE={metrics['RMSE']:.4f}  "
              f"MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.4f}")


if __name__ == "__main__":
    main()
