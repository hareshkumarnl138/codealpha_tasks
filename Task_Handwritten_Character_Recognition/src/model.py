import torch.nn as nn

class CharacterCNN(nn.Module):
    """Compact CNN for 28x28 grayscale EMNIST characters."""
    def __init__(self, num_classes=62):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.35), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))
