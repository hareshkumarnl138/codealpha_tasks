
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

DATA_DIR = Path("data/crema_d")
AUDIO_DIR = DATA_DIR / "audio"
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_FRAMES = 130

metadata = pd.read_csv(DATA_DIR / "metadata.csv")

# Keep only clips whose audio files exist
metadata["path"] = metadata["filename"].apply(
    lambda name: AUDIO_DIR / name
)
metadata = metadata[metadata["path"].apply(Path.exists)].reset_index(drop=True)

print("Valid audio files:", len(metadata))

# Encode emotion labels
encoder = LabelEncoder()
metadata["label"] = encoder.fit_transform(metadata["emotion_code"])

X = []
y = []
groups = []
valid_rows = []

for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting MFCCs"):
    try:
        audio, _ = librosa.load(
            str(row["path"]),
            sr=SAMPLE_RATE,
            mono=True,
            duration=3.0
        )

        # Pad short recordings and trim long recordings
        if len(audio) < SAMPLE_RATE * 3:
            audio = np.pad(audio, (0, SAMPLE_RATE * 3 - len(audio)))
        else:
            audio = audio[:SAMPLE_RATE * 3]

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=2048,
            hop_length=512
        )

        # Standardize each coefficient across time
        mean = mfcc.mean(axis=1, keepdims=True)
        std = mfcc.std(axis=1, keepdims=True) + 1e-6
        mfcc = (mfcc - mean) / std

        # Make every feature matrix the same size
        if mfcc.shape[1] < MAX_FRAMES:
            mfcc = np.pad(
                mfcc,
                ((0, 0), (0, MAX_FRAMES - mfcc.shape[1]))
            )
        else:
            mfcc = mfcc[:, :MAX_FRAMES]

        X.append(mfcc.astype(np.float32))
        y.append(row["label"])
        groups.append(str(row["actor_id"]))
        valid_rows.append(row["filename"])

    except Exception as exc:
        print(f"\nSkipping {row['filename']}: {exc}")

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int64)
groups = np.asarray(groups)

# Split by actor to reduce speaker leakage.
# First reserve 20% of actors, then split that group into validation/test.
outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, hold_idx = next(outer.split(X, y, groups))

inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_rel, test_rel = next(
    inner.split(X[hold_idx], y[hold_idx], groups[hold_idx])
)
val_idx = hold_idx[val_rel]
test_idx = hold_idx[test_rel]

np.savez_compressed(
    OUTPUT_DIR / "crema_mfcc.npz",
    X_train=X[train_idx], y_train=y[train_idx],
    X_val=X[val_idx], y_val=y[val_idx],
    X_test=X[test_idx], y_test=y[test_idx],
    classes=encoder.classes_
)

print("\nPreprocessing complete!")
print("Classes:", list(encoder.classes_))
print("Train:", len(train_idx))
print("Validation:", len(val_idx))
print("Test:", len(test_idx))
print("Saved:", OUTPUT_DIR / "crema_mfcc.npz")
