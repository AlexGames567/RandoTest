import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Bildklassifikation",
    page_icon="🖼️",
    layout="centered"
)

MODEL_PATH = "keras_model.h5"
CLASS_NAMES_PATH = "class_names.txt"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
        )


def load_class_names():
    try:
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        return None


def preprocess_image(image, target_size):
    image = image.convert("RGB")
    image = image.resize(target_size)

    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def get_prediction(model, image_array, class_names=None):
    prediction = model.predict(image_array, verbose=0)
    prediction = np.asarray(prediction)

    # Binäre Klassifikation mit einem einzelnen Ausgang
    if prediction.size == 1:
        probability = float(prediction.flatten()[0])

        if class_names and len(class_names) >= 2:
            if probability >= 0.5:
                class_index = 1
                confidence = probability
            else:
                class_index = 0
                confidence = 1.0 - probability

            label = class_names[class_index]
        else:
            label = "Klasse 1" if probability >= 0.5 else "Klasse 0"
            confidence = probability if probability >= 0.5 else 1.0 - probability

        return label, confidence

    # Mehrklassige Klassifikation
    probabilities = prediction.flatten()

    class_index = int(np.argmax(probabilities))
    confidence = float(probabilities[class_index])

    if class_names and class_index < len(class_names):
        label = class_names[class_index]
    else:
        label = f"Klasse {class_index}"

    return label, confidence


st.title("🖼️ Bildklassifikation mit Keras")
st.write("Lade ein Bild hoch und lass das trainierte Modell erkennen, was darauf zu sehen ist.")

try:
    model = load_model()
except Exception as error:
    st.error(f"Das Modell konnte nicht geladen werden:\n\n{error}")
    st.stop()

class_names = load_class_names()

# Eingabegröße des Modells automatisch auslesen
input_shape = model.input_shape

if isinstance(input_shape, list):
    input_shape = input_shape[0]

if len(input_shape) != 4 or input_shape[1] is None or input_shape[2] is None:
    st.error(
        "Die Eingabegröße des Modells konnte nicht automatisch erkannt werden. "
        "Erwartet wird zum Beispiel ein Modell mit der Form (None, 224, 224, 3)."
    )
    st.stop()

image_width = int(input_shape[2])
image_height = int(input_shape[1])

st.caption(f"Erkannte Modell-Bildgröße: {image_width} × {image_height} Pixel")

uploaded_file = st.file_uploader(
    "Bild auswählen",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Hochgeladenes Bild",
        use_container_width=True
    )

    image_array = preprocess_image(
        image,
        target_size=(image_width, image_height)
    )

    if st.button("Bild analysieren", type="primary"):
        with st.spinner("Das Bild wird analysiert ..."):
            label, confidence = get_prediction(
                model,
                image_array,
                class_names
            )

        st.success(f"Das Bild zeigt vermutlich: **{label}**")
        st.metric(
            "Wahrscheinlichkeit",
            f"{confidence * 100:.2f} %"
        )

        if confidence < 0.6:
            st.warning(
                "Die Vorhersage ist relativ unsicher. "
                "Das Bild unterscheidet sich möglicherweise von den Trainingsdaten."
            )
