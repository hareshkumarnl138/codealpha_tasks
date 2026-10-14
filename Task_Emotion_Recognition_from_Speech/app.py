
from pathlib import Path
import tempfile

import librosa
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="SpeechSense AI | Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("models/crema_cnn_bilstm.pt")
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_FRAMES = 130

EMOTION_NAMES = {
    "ANG": "Anger",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad",
}

EMOTION_ICONS = {
    "ANG": "😠",
    "DIS": "🤢",
    "FEA": "😨",
    "HAP": "😊",
    "NEU": "😐",
    "SAD": "😢",
}

EMOTION_COLORS = {
    "ANG": "#EF4444",
    "DIS": "#A855F7",
    "FEA": "#F59E0B",
    "HAP": "#10B981",
    "NEU": "#64748B",
    "SAD": "#3B82F6",
}


# --------------------------------------------------
# Model architecture (must match training)
# --------------------------------------------------
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

        self.lstm = nn.LSTM(
            input_size=320,
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
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)
        return self.classifier(x[:, -1, :])


# --------------------------------------------------
# Load model once
# --------------------------------------------------
@st.cache_resource
def load_emotion_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Train the model first using train_crema.py."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    classes = checkpoint["classes"]
    model = CNNBiLSTM(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, classes, device


# --------------------------------------------------
# Audio preprocessing (same as training)
# --------------------------------------------------
def extract_features(audio_path):
    audio, _ = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        mono=True,
        duration=3.0,
    )

    if len(audio) == 0:
        raise ValueError("The uploaded audio contains no readable samples.")

    target_length = SAMPLE_RATE * 3

    if len(audio) < target_length:
        audio = np.pad(
            audio,
            (0, target_length - len(audio)),
        )
    else:
        audio = audio[:target_length]

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=2048,
        hop_length=512,
    )

    mean = mfcc.mean(axis=1, keepdims=True)
    std = mfcc.std(axis=1, keepdims=True) + 1e-6
    mfcc = (mfcc - mean) / std

    if mfcc.shape[1] < MAX_FRAMES:
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, MAX_FRAMES - mfcc.shape[1])),
        )
    else:
        mfcc = mfcc[:, :MAX_FRAMES]

    return torch.tensor(
        mfcc,
        dtype=torch.float32,
    ).unsqueeze(0)


def predict_emotion(audio_path, model, classes, device):
    features = extract_features(audio_path).to(device)

    with torch.no_grad():
        logits = model(features)
        probabilities = F.softmax(logits, dim=1)[0].cpu().numpy()

    ranking = np.argsort(probabilities)[::-1]
    return probabilities, ranking


# --------------------------------------------------
# Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }
    .hero {
        padding: 1.7rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #10243A, #174C5B);
        color: white;
        margin-bottom: 1.4rem;
    }
    .hero h1 {
        color: white;
        margin-bottom: 0.4rem;
        font-size: 2.25rem;
    }
    .hero p {
        color: #D5E8EF;
        font-size: 1rem;
        margin-bottom: 0;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        min-height: 180px;
    }
    .result-emoji {
        font-size: 2.8rem;
    }
    .result-name {
        font-size: 1.8rem;
        font-weight: 750;
        margin-top: 0.4rem;
    }
    .muted {
        color: #94A3B8;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("🎙️ SpeechSense AI")
    st.caption("Speech Emotion Recognition")
    st.divider()

    st.subheader("About the model")
    st.write("**Architecture:** CNN + BiLSTM")
    st.write("**Audio features:** 40 MFCC coefficients")
    st.write("**Dataset:** CREMA-D")
    st.write("**Emotion classes:** 6")

    st.divider()
    st.subheader("Supported emotions")

    for code in ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]:
        st.write(f"{EMOTION_ICONS[code]} {EMOTION_NAMES[code]}")

    st.divider()
    st.caption(
        "Research and educational demonstration. "
        "Predictions are estimates and may be incorrect."
    )


# --------------------------------------------------
# Main interface
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Understand the Emotion Behind the Voice</h1>
        <p>
            Upload a speech recording and let AI estimate its
            emotional expression using deep learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.subheader("1. Upload your audio")
    st.write("Choose a clear speech recording in WAV format.")

    uploaded_file = st.file_uploader(
        "Select an audio file",
        type=["wav"],
        help="Use a WAV file with audible speech, preferably a few seconds long.",
    )

    if uploaded_file is not None:
        st.audio(uploaded_file.getvalue(), format="audio/wav")

        st.caption(
            f"File: {uploaded_file.name} · "
            f"{uploaded_file.size / 1024:.1f} KB"
        )

    analyze = st.button(
        "Analyze Emotion",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

with right:
    st.subheader("How it works")

    st.markdown(
        """
        <div style="padding: 1rem; border-radius: 12px;
                    border: 1px solid rgba(128,128,128,0.25);">
          <p><b>01 · Audio input</b><br>
          <span class="muted">Upload a speech recording.</span></p>
          <p><b>02 · Feature extraction</b><br>
          <span class="muted">Convert speech into MFCC features.</span></p>
          <p><b>03 · Deep learning</b><br>
          <span class="muted">CNN + BiLSTM processes the features.</span></p>
          <p style="margin-bottom:0"><b>04 · Emotion estimate</b><br>
          <span class="muted">View the leading class and all scores.</span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Inference and results
# --------------------------------------------------
if analyze and uploaded_file is not None:
    with st.spinner("Analyzing speech..."):
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = Path(temp_file.name)

            model, classes, device = load_emotion_model()
            probabilities, ranking = predict_emotion(
                temp_path,
                model,
                classes,
                device,
            )

            top_index = int(ranking[0])
            top_code = classes[top_index]
            top_name = EMOTION_NAMES.get(top_code, top_code)
            top_score = float(probabilities[top_index])

            st.divider()
            st.subheader("2. Emotion analysis results")

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Predicted emotion",
                f"{EMOTION_ICONS.get(top_code, '🎙️')} {top_name}",
            )
            metric2.metric(
                "Top model score",
                f"{top_score:.1%}",
            )
            metric3.metric(
                "Audio duration analyzed",
                "Up to 3 seconds",
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-emoji">
                        {EMOTION_ICONS.get(top_code, '🎙️')}
                    </div>
                    <div class="result-name">{top_name}</div>
                    <p class="muted">
                        Highest-scoring predicted emotion
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.subheader("Emotion score comparison")

            chart_data = pd.DataFrame({
                "Emotion": [
                    EMOTION_NAMES.get(code, code)
                    for code in classes
                ],
                "Model score (%)": probabilities * 100,
            }).sort_values("Model score (%)", ascending=False)

            st.bar_chart(
                chart_data.set_index("Emotion"),
                horizontal=True,
                x_label="Model score (%)",
                y_label="Emotion",
            )

            st.subheader("Detailed scores")

            for index in ranking:
                code = classes[int(index)]
                name = EMOTION_NAMES.get(code, code)
                score = float(probabilities[int(index)])

                st.write(
                    f"{EMOTION_ICONS.get(code, '🎙️')} "
                    f"**{name}** — {score:.1%}"
                )
                st.progress(min(max(score, 0.0), 1.0))

            st.info(
                "These percentages are softmax model scores, not "
                "verified probabilities that a person truly feels "
                "the emotion. Speech emotion recognition is uncertain "
                "and can be affected by the speaker, language, "
                "recording quality and context."
            )

        except Exception as exc:
            st.error(f"Could not analyze the audio: {exc}")
            st.caption(
                "Check that the trained model exists at "
                "models/crema_cnn_bilstm.pt and the WAV file is valid."
            )

        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

st.divider()
st.caption(
    "SpeechSense AI · CodeAlpha Task 2 · "
    "Emotion Recognition from Speech"
)
