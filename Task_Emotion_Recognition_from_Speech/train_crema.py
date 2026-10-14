
from pathlib import Path
import copy
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt

DATA_FILE = Path("data/processed/crema_mfcc.npz")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
PATIENCE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", device)

data = np.load(DATA_FILE, allow_pickle=True)
classes = data["classes"].tolist()
num_classes = len(classes)

print("Emotion classes:", classes)


class CNNBiLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )

        # 40 MFCC coefficients -> 5 frequency positions
        self.lstm = nn.LSTM(
            input_size=64 * 5,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # Input: [batch, 40 MFCCs, 130 frames]
        x = x.unsqueeze(1)
        x = self.cnn(x)
        # [batch, channels, frequency, time]
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.classifier(x)


def make_loader(x, y, shuffle=False):
    dataset = TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y).long()
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


train_loader = make_loader(
    data["X_train"], data["y_train"], shuffle=True
)
val_loader = make_loader(data["X_val"], data["y_val"])
test_loader = make_loader(data["X_test"], data["y_test"])

model = CNNBiLSTM(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

best_val_loss = float("inf")
best_state = copy.deepcopy(model.state_dict())
wait = 0

history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
}

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    train_correct = 0
    train_total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        train_correct += (logits.argmax(1) == y).sum().item()
        train_total += y.size(0)

    train_loss = total_loss / train_total
    train_acc = train_correct / train_total

    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            val_loss_sum += loss.item() * y.size(0)
            val_correct += (logits.argmax(1) == y).sum().item()
            val_total += y.size(0)

    val_loss = val_loss_sum / val_total
    val_acc = val_correct / val_total

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_accuracy"].append(train_acc)
    history["val_accuracy"].append(val_acc)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: {train_acc:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val accuracy: {val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        wait = 0

        torch.save(
            {
                "model_state_dict": best_state,
                "classes": classes,
            },
            MODEL_DIR / "crema_cnn_bilstm.pt",
        )
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping: validation loss did not improve.")
            break

# Evaluate the best validation checkpoint on the test set
model.load_state_dict(best_state)
model.eval()

all_true = []
all_pred = []

with torch.no_grad():
    for x, y in test_loader:
        logits = model(x.to(device))
        predictions = logits.argmax(1).cpu().numpy()
        all_pred.extend(predictions.tolist())
        all_true.extend(y.numpy().tolist())

accuracy = accuracy_score(all_true, all_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_true,
    all_pred,
    average="macro",
    zero_division=0,
)

metrics = {
    "test_accuracy": float(accuracy),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(f1),
    "classes": classes,
    "test_samples": len(all_true),
}

print("\nTEST RESULTS")
print(json.dumps(metrics, indent=2))
print("\nClassification report:")
print(classification_report(
    all_true,
    all_pred,
    labels=list(range(num_classes)),
    target_names=classes,
    zero_division=0,
))

with open(RESULTS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open(RESULTS_DIR / "training_history.json", "w") as f:
    json.dump(history, f, indent=2)

cm = confusion_matrix(
    all_true, all_pred, labels=list(range(num_classes))
)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation="nearest")
ax.set(
    xticks=np.arange(num_classes),
    yticks=np.arange(num_classes),
    xticklabels=classes,
    yticklabels=classes,
    xlabel="Predicted emotion",
    ylabel="Actual emotion",
    title="CREMA-D Test Confusion Matrix",
)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["train_loss"], label="Train")
axes[0].plot(history["val_loss"], label="Validation")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history["train_accuracy"], label="Train")
axes[1].plot(history["val_accuracy"], label="Validation")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()

fig.tight_layout()
fig.savefig(RESULTS_DIR / "training_curves.png", dpi=160)
plt.close(fig)

print("\nTraining complete.")
print("Model: models/crema_cnn_bilstm.pt")
print("Metrics: results/metrics.json")
print("Confusion matrix: results/confusion_matrix.png")
print("Training curves: results/training_curves.png")
