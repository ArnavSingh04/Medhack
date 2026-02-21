"""
LSTM model to predict state label from vital signs (row-level classification).
Maximizes competition expected score.

Output: predictions.csv with columns: ID, predicted_label
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# CONFIG
# =========================
BATCH_SIZE = 256
EPOCHS = 15
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_COLS = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "oxygen_saturation",
]

TARGET_COL = "label"

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_PATH = DATA_DIR / "train_data.csv"
TEST_PATH = DATA_DIR / "holdout_data.csv"
OUTPUT_PATH = DATA_DIR / "predictions.csv"

NUM_CLASSES = 4

# =========================
# COMPETITION SCORING MATRIX
# =========================
SCORE_MATRIX = torch.tensor([
    [0,  -2,  -2,  -5],
    [-3,  2,  -1,  -5],
    [-10, -3,  3,  -5],
    [-15, -10, -5,  5]
], dtype=torch.float32).to(DEVICE)

# =========================
# DATASET
# =========================
class RowDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# =========================
# MODEL
# =========================
class SimpleMLP(nn.Module):
    def __init__(self, input_size=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# CUSTOM COMPETITION LOSS
# =========================
class CompetitionLoss(nn.Module):
    def __init__(self, score_matrix):
        super().__init__()
        self.score_matrix = score_matrix

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        score_rows = self.score_matrix[targets]
        expected_score = torch.sum(probs * score_rows, dim=1)
        return -expected_score.mean()

# =========================
# TRAINING LOOP
# =========================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# =========================
# MAIN
# =========================
def main():

    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Remove rows with all features null
    train_df = train_df[~train_df[FEATURE_COLS].isnull().all(axis=1)]
    test_df = test_df[~test_df[FEATURE_COLS].isnull().all(axis=1)]

    print("Raw label distribution:")
    print(train_df[TARGET_COL].value_counts())

    # Impute
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train_df[FEATURE_COLS])
    X_test = imputer.transform(test_df[FEATURE_COLS])

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = train_df[TARGET_COL].astype(int).values

    print("Processed class distribution:",
          np.bincount(y_train, minlength=NUM_CLASSES))

    train_dataset = RowDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleMLP(input_size=len(FEATURE_COLS)).to(DEVICE)
    criterion = CompetitionLoss(SCORE_MATRIX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Training...")
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS} - Expected Score Loss: {loss:.4f}")

    print("Predicting test set...")
    model.eval()

    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        logits = model(test_tensor)
        probs = torch.softmax(logits, dim=1)

        expected_scores = probs @ SCORE_MATRIX.T
        predicted = torch.argmax(expected_scores, dim=1).cpu().numpy()

    predictions_df = pd.DataFrame({
        "ID": np.arange(1, len(predicted) + 1),
        "predicted_label": predicted
    })

    predictions_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved predictions with {len(predictions_df)} rows.")
    print("Prediction distribution:",
          dict(zip(*np.unique(predicted, return_counts=True))))


if __name__ == "__main__":
    main()