# CodeAlpha Task 2 — Emotion Recognition from Speech

An end-to-end speech emotion classification project using MFCC audio features and a CNN + bidirectional LSTM model. Includes dataset checks, feature extraction, actor-independent evaluation, CLI prediction, and a Streamlit demo.

## Classes
RAVDESS labels: neutral, calm, happy, sad, angry, fearful, disgust, surprised.

## Project structure

```text
Task_Emotion_Recognition_from_Speech/
├── data/raw/                 # Local RAVDESS dataset (not committed)
├── data/processed/           # Extracted features (not committed)
├── models/                   # Model weights (not committed)
├── results/figures/
├── results/tables/
├── src/                      # preprocessing, training, prediction
├── app.py
├── requirements.txt
└── README.md
```

## Windows setup

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset setup
Download the RAVDESS speech audio dataset from [RAVDESS on Zenodo](https://zenodo.org/records/1188976). Extract the speech WAV archive so `data/raw/` contains `Actor_01` through `Actor_24`, each with `.wav` files. Use the speech audio archive, not song-only files.

## Run in order

```bat
python -m src.01_check_dataset
python -m src.02_prepare_data
python -m src.03_train_model
```

Then predict an audio file:

```bat
python -m src.04_predict "path\to\speech.wav"
```

Launch the upload interface:

```bat
streamlit run app.py
```

## Methodology
- Librosa loads mono audio at 22.05 kHz and pads/trims to three seconds.
- Extracts 40 MFCC coefficients and normalizes each recording's coefficients.
- Uses an actor-independent split: actors 01–18 train, 19–21 validation, and 22–24 test. This helps limit speaker leakage.
- Trains a CNN + bidirectional LSTM with early stopping and evaluates once on the held-out test actors.
- Outputs metrics JSON, per-class classification report, confusion matrix, and loss curve.

Do not report model accuracy until you have trained it on the real dataset. Performance depends on the data and run configuration.

## Responsible use
Emotion labels inferred from speech are uncertain and affected by acting, accent, language, recording quality, and individual differences. This educational model cannot establish a person's true internal feelings and should not be used for high-stakes decisions.

## Attribution
RAVDESS is the Ryerson Audio-Visual Database of Emotional Speech and Song. Follow the dataset's official terms and attribution requirements.
