
from PIL import Image
import numpy as np
import streamlit as st
import torch

from src.data_utils import ROOT, image_to_tensor
from src.model import CharacterCNN


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="InkSense | Character Recognition",
    page_icon="✍️",
    layout="wide",
)


# --------------------------------------------------
# 2. CUSTOM STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background: #f7f9fc;
    }

    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #14213d, #284b63);
        color: white;
        margin-bottom: 1rem;
    }

    .hero h1 {
        color: white;
        margin: 0;
    }

    .hero p {
        color: #e8eef5;
        margin: .4rem 0 0;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid #e6eaf0;
    }
    </style>

    <div class="hero">
        <h1>✍️ InkSense</h1>
        <p>
            Handwritten character recognition powered by a CNN
            trained on EMNIST ByClass.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. MODEL CONFIGURATION
# --------------------------------------------------
MODEL_PATH = ROOT / "models" / "emnist_cnn.pth"


# --------------------------------------------------
# 4. LOAD TRAINED MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None, None

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    classes = checkpoint["classes"]

    model = CharacterCNN(len(classes)).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, classes, device


model, classes, device = load_model()


if model is None:
    st.warning(
        "The trained model is not available yet. "
        "In CMD, run: python train.py --epochs 8 --batch-size 128"
    )
    st.stop()


# --------------------------------------------------
# 5. INPUT AND PREDICTION LAYOUT
# --------------------------------------------------
left, right = st.columns(
    [1.05, 0.95],
    gap="large",
)


# --------------------------------------------------
# 6. INPUT SECTION
# --------------------------------------------------
with left:
    st.subheader("Choose your input")

    mode = st.radio(
        "Input method",
        ["Draw a character", "Upload an image"],
        horizontal=True,
    )

    input_image = None

    if mode == "Draw a character":
        st.caption(
            "Draw one large character in the white box "
            "using a dark stroke."
        )

        try:
            from streamlit_drawable_canvas import st_canvas

            canvas = st_canvas(
                fill_color="rgba(255,255,255,0)",
                stroke_width=18,
                stroke_color="#111111",
                background_color="#FFFFFF",
                height=300,
                width=300,
                drawing_mode="freedraw",
                key="ink_canvas",

                # FIX: Request canvas image data
                return_image_data=True,
            )

            # Read the drawing when the canvas contains a stroke.
            if (
                canvas.image_data is not None
                and np.any(canvas.image_data[:, :, :3] < 245)
            ):
                canvas_array = np.clip(
                    canvas.image_data, 0, 255
                ).astype(np.uint8)

                input_image = Image.fromarray(
                    canvas_array, mode="RGBA"
                ).convert("RGB")

        except ImportError:
            st.error(
                "Canvas package is missing. "
                "Install it using: "
                "python -m pip install streamlit-drawable-canvas"
            )

    else:
        uploaded = st.file_uploader(
            "Upload one handwritten character",
            type=["png", "jpg", "jpeg"],
        )

        if uploaded is not None:
            try:
                input_image = Image.open(uploaded).convert("RGB")

                st.image(
                    input_image,
                    caption="Uploaded image",
                    width=280,
                )

            except Exception as error:
                st.error(f"Unable to open the image: {error}")

    clicked = st.button(
        "Recognize character",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# 7. PREDICTION SECTION
# --------------------------------------------------
with right:
    st.subheader("Prediction")

    if clicked:
        if input_image is None:
            st.error(
                "Please draw a character or upload an image first."
            )

        else:
            try:
                # Apply the project's image preprocessing.
                x = image_to_tensor(input_image).to(device)

                with torch.no_grad():
                    logits = model(x)
                    probabilities = torch.softmax(logits, dim=1)[0]

                    top_k = min(5, len(classes))
                    values, indices = torch.topk(
                        probabilities, k=top_k
                    )

                predicted_index = indices[0].item()
                predicted_character = str(classes[predicted_index])
                confidence = values[0].item() * 100

                st.metric(
                    "Predicted character",
                    predicted_character,
                    f"{confidence:.1f}% model score",
                )

                st.caption(
                    "The model score is not a guarantee of correctness."
                )

                st.markdown("**Top 5 predictions**")

                for probability, index in zip(values, indices):
                    character = str(classes[index.item()])
                    score = probability.item()

                    st.write(
                        f"`{character}` — {score * 100:.2f}%"
                    )

                    st.progress(float(score))

                # Display the processed image given to the model.
                with st.expander("View processed 28×28 input"):
                    processed_tensor = image_to_tensor(input_image)

                    processed_array = (
                        processed_tensor[0, 0]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    shown = np.clip(
                        (processed_array + 1) * 127.5,
                        0,
                        255,
                    ).astype(np.uint8)

                    st.image(
                        Image.fromarray(shown).resize(
                            (224, 224)
                        ),
                        caption="Normalized model input",
                    )

            except Exception as error:
                st.error(
                    f"Prediction failed: {error}"
                )

    else:
        st.info(
            "Draw or upload a single character, "
            "then click Recognize character."
        )


# --------------------------------------------------
# 8. ABOUT THE PROJECT
# --------------------------------------------------
st.divider()

st.markdown(
    """
    **About:** InkSense predicts one handwritten character per image:
    digits 0–9 and uppercase/lowercase letters.

    It does not recognize complete words or sentences.
    """
)


# --------------------------------------------------
# 9. MODEL AND DATASET DETAILS
# --------------------------------------------------
with st.expander("Model and dataset details"):
    st.write(
        "Dataset: EMNIST ByClass (62 classes). "
        "Model: Convolutional Neural Network (CNN). "
        "Input: 28×28 grayscale image. "
        "Evaluation metrics are generated locally after training."
    )

    st.write(
        f"Device: {device}"
    )

    st.write(
        f"Number of character classes: {len(classes)}"
    )
