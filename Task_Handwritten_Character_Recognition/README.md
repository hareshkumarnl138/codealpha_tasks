# CodeAlpha Task 3 — Handwritten Character Recognition

A complete local project recognizing handwritten digits and letters using **EMNIST ByClass (62 classes)** and a PyTorch Convolutional Neural Network (CNN). It includes preprocessing, training, evaluation, CLI prediction, and a Streamlit drawing/upload interface.

## Features
- EMNIST ByClass: digits 0–9, uppercase A–Z, lowercase a–z
- CNN with batch normalization, dropout, AdamW and learning-rate scheduling
- Stratified 90/10 train/validation split from the official training set; official test set kept for final evaluation
- Test metrics, classification report, confusion matrix, training curves
- Streamlit interface: draw one character or upload a PNG/JPG
- CLI prediction with top-five class scores

## Setup (Windows CMD)
Python 3.10–3.12 recommended. Open CMD in this folder:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train
For a quick pipeline test:

```bat
python train.py --epochs 3 --batch-size 128
```

For a longer run:

```bat
python train.py --epochs 8 --batch-size 128
```

The first run downloads EMNIST into `data/`. Training can take a while on CPU; an NVIDIA GPU can speed it up. The best validation checkpoint is saved as `models/emnist_cnn.pth`. Metrics and plots are saved in `results/`.

## Run the app
After training completes:

```bat
streamlit run app.py
```

Open the local URL printed in CMD (usually `http://localhost:8501`). Select **Draw a character** or **Upload an image**, then click **Recognize character**.

## Predict from CMD

```bat
python predict.py "sample_character.png"
```

Use a clear image with one centered character. This is single-character classification, not whole-word OCR. User-drawn or uploaded images can differ from the EMNIST training distribution, so predictions may be less reliable for unusual handwriting.

## Evaluation honesty
The actual test metrics are generated on your computer and saved to `results/metrics.json`. Do not claim an accuracy until your training run completes.
