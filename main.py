import os

# TensorFlow-Meldungen reduzieren
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# --------------------------------------------------
# Einstellungen
# --------------------------------------------------

MODEL_PATH = "keras_model.h5"
CLASS_NAMES_PATH = "class_names.txt"


# --------------------------------------------------
# Kompatible DepthwiseConv2D-Klasse
# Behebt den Fehler mit "groups": 1
# --------------------------------------------------

class CompatibleDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, groups=1, **kwargs):
        # Ältere TensorFlow-Versionen unterstützen "groups"
        # bei DepthwiseConv2D nicht.
        super().__init__(*args, **kwargs)


# --------------------------------------------------
# Modell laden
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            "DepthwiseConv2D": CompatibleDepthwiseConv2D
        }
    )


# --------------------------------------------------
# Klassennamen laden
# --------------------------------------------------

def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        return None

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        names = [
            line.strip()
            for line in file.readlines()
            if line.strip()
        ]

    return names if names else None


# --------------------------------------------------
# Bild vorbereiten
# --------------------------------------------------

def preprocess_image(image, width, height):
    image = image.convert("RGB")
    image = image.resize((width, height))

    image_array = np.asarray(image).astype("float32")

    # Standard-Normalisierung für viele Keras-Bildmodelle
    image_array = image_array / 255.0

    # Batch-Dimension hinzufügen:
    # aus (H, W, 3) wird (1, H, W, 3)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# --------------------------------------------------
# Vorhersage
# --------------------------------------------------

def predict_image(model, image_array, class_names):
    prediction = model.predict(image_array, verbose=0)
    prediction = np.asarray(prediction).flatten()

    # Binäre Klassifikation:
    # Das Modell gibt einen einzelnen Wert zwischen 0 und 1 aus.
    if len(prediction) == 1:
        probability_class_1 = float(prediction[0])

        if probability_class_1 >= 0.5:
            class_index = 1
            confidence = probability_class_1
        else:
            class_index = 0
            confidence = 1.0 - probability_class_1

        if class_names and len(class_names) >= 2:
            label = class_names[class_index]
        else:
            label = f"Klasse {class_index}"

        return label, confidence

    # Mehrklassige Klassifikation
    class_index = int(np.argmax(prediction))
    confidence = float(prediction[class_index])

    if class_names and class_index < len(class_names):
        label = class_names[class_index]
    else:
        label = f"Klasse {class_index}"

    return label, confidence


# --------------------------------------------------
# Streamlit-Oberfläche
# --------------------------------------------------

st.set_page_config(
    page_title="Bildklassifikation",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Bildklassifikation")
st.write(
    "Lade ein Bild hoch. Das trainierte Keras-Modell "
    "versucht anschließend zu erkennen, was darauf zu sehen ist."
)


# Modell laden
try:
    model = load_model()
except Exception as error:
    st.error("Das Modell konnte nicht geladen werden.")
    st.code(str(error))

    st.info(
        "Überprüfe, ob sich die Datei 'keras_model.h5' "
        "im gleichen Ordner wie 'app.py' befindet."
    )

    st.stop()


# Klassennamen laden
class_names = load_class_names()


# Eingabegröße des Modells bestimmen
try:
    input_shape = model.input_shape

    # Bei Modellen mit mehreren Eingaben
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) != 4:
        st.error(
            f"Nicht unterstützte Eingabeform des Modells: {input_shape}"
        )
        st.stop()

    _, image_height, image_width, channels = input_shape

    if image_height is None or image_width is None:
        st.error(
            "Die Bildgröße des Modells konnte nicht automatisch erkannt werden."
        )
        st.stop()

    image_height = int(image_height)
    image_width = int(image_width)

except Exception as error:
    st.error(f"Die Eingabegröße des Modells konnte nicht gelesen werden: {error}")
    st.stop()


st.caption(
    f"Modell-Bildgröße: {image_width} × {image_height} Pixel"
)


# Bild hochladen
uploaded_file = st.file_uploader(
    "Bild auswählen",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Hochgeladenes Bild",
            use_container_width=True
        )

        if st.button("Bild analysieren", type="primary"):
            with st.spinner("Bild wird analysiert ..."):
                image_array = preprocess_image(
                    image,
                    width=image_width,
                    height=image_height
                )

                label, confidence = predict_image(
                    model,
                    image_array,
                    class_names
                )

            st.success(
                f"Das Bild zeigt vermutlich: {label}"
            )

            st.metric(
                label="Sicherheit",
                value=f"{confidence * 100:.2f} %"
            )

            if confidence < 0.60:
                st.warning(
                    "Die Vorhersage ist unsicher. "
                    "Das Bild ist möglicherweise nicht ähnlich "
                    "zu den Trainingsbildern."
                )

    except Exception as error:
        st.error(f"Das Bild konnte nicht verarbeitet werden: {error}")
