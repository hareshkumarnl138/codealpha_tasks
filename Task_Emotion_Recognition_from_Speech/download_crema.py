
from datasets import load_dataset
from pathlib import Path
import pandas as pd
import soundfile as sf

OUTPUT = Path("data/crema_d")
AUDIO_DIR = OUTPUT / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading CREMA-D from Hugging Face...")
dataset = load_dataset("cfahlgren1/crema-d", split="train")

metadata = []

for i, row in enumerate(dataset):
    audio = row["audio"]
    label = str(row["emotion_code"]).upper()

    # Save the decoded audio as a WAV file
    filename = f"clip_{i:05d}_{label}.wav"
    sf.write(
        AUDIO_DIR / filename,
        audio["array"],
        audio["sampling_rate"]
    )

    metadata.append({
        "filename": filename,
        "emotion_code": label,
        "actor_id": row.get("actor_id", ""),
        "sentence_code": row.get("sentence_code", "")
    })

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1}/{len(dataset)} clips")

pd.DataFrame(metadata).to_csv(
    OUTPUT / "metadata.csv", index=False
)

print(f"\nDone! Saved {len(metadata)} audio clips.")
print(f"Audio folder: {AUDIO_DIR}")
print(f"Metadata: {OUTPUT / 'metadata.csv'}")
print(pd.DataFrame(metadata)["emotion_code"].value_counts())
