
from pathlib import Path
import sys

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_FRAMES = 130
MODEL_PATH = Path("models/crema_cnn_bilstm.pt")


class CNNBiLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )
        self.lstm = nn.LSTM(
            320, 128, batch_first=True, bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x.unsqueeze(1))
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)
        return self.classifier(x[:, -1, :])


if len(sys.argv) < 2:
    print('Usage: python predict_crema.py "path\\to\\audio.wav"')
    raise SystemExit(1)

audio_path = Path(sys.argv[1])
if not audio_path.is_file():
    raise FileNotFoundError(f"Audio file not found: {audio_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(
    MODEL_PATH, map_location=device, weights_only=True
)
classes = checkpoint["classes"]

model = CNNBiLSTM(len(classes)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

audio, _ = librosa.load(
    str(audio_path), sr=SAMPLE_RATE, mono=True, duration=3.0
)

target_length = SAMPLE_RATE * 3
if len(audio) < target_length:
    audio = np.pad(audio, (0, target_length - len(audio)))
else:
    audio = audio[:target_length]

mfcc = librosa.feature.mfcc(
    y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
    n_fft=2048, hop_length=512
)
mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (
    mfcc.std(axis=1, keepdims=True) + 1e-6
)

if mfcc.shape[1] < MAX_FRAMES:
    mfcc = np.pad(mfcc, ((0, 0), (0, MAX_FRAMES - mfcc.shape[1])))
else:
    mfcc = mfcc[:, :MAX_FRAMES]

x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    probabilities = F.softmax(model(x), dim=1)[0]

ranking = torch.argsort(probabilities, descending=True)
emotion_names = {
    "ANG": "Anger",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad",
}

print(f"\nAudio: {audio_path.name}")
print(f"Predicted emotion: {emotion_names.get(classes[ranking[0]], classes[ranking[0]])}")
print(f"Model score: {probabilities[ranking[0]].item():.1%}")
print("\nAll class scores:")

for index in ranking:
    label = classes[index]
    print(f"{emotion_names.get(label, label):10s}: {probabilities[index].item():.1%}")
