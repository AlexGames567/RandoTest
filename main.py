
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "keras_model.h5"
CLASS_NAMES_PATH = "labels.txt"


class CompatibleDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    """
    Kompatibilität für ältere Keras-Modelle.
    """
    def __init__(self, *args, **kwargs):
        # Entferne 'groups' falls es zu Konflikten führt
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)

@st.cache_resource
def load_model():
    # Verwende den custom_object_scope, um Keras beim Deserialisieren zu helfen
    custom_objects = {"CompatibleDepthwiseConv2D": CompatibleDepthwiseConv2D, "DepthwiseConv2D": CompatibleDepthwiseConv2D}
    
    with tf.keras.utils.custom_object_scope(custom_objects):
        try:
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            raise RuntimeError(f"Fehler beim Laden des Modells: {e}")


def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        return None

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def preprocess_image(image, width, height):
    image = image.convert("RGB")
    image = image.resize((width, height))

    image_array = np.asarray(image).astype("float32")

    # Standard für viele Keras-Modelle (Normalisierung auf [0, 1])
    image_array = image_array / 255.0

    # Form: (1, Höhe, Breite, 3)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def make_prediction(model, image_array, class_names):
    prediction = model.predict(
        image_array,
        batch_size=1,
        verbose=0
    )

    prediction = np.asarray(prediction).flatten()

    # Binäre Klassifikation mit einem Ausgang
    if prediction.size == 1:
        probability = float(prediction[0])

        if probability >= 0.5:
            class_index = 1
            confidence = probability
        else:
            class_index = 0
            confidence = 1.0 - probability

    # Mehrklassige Klassifikation
    else:
        class_index = int(np.argmax(prediction))
        confidence = float(prediction[class_index])

    if class_names and class_index < len(class_names):
        label = class_names[class_index]
    else:
        label = f"Klasse {class_index}"

    return label, confidence


st.set_page_config(
    page_title="Bildklassifikation",
    page_icon="🖼️"
)

st.title("🖼️ Bildklassifikation")

try:
    model = load_model()
except Exception as error:
    st.error("Das Modell konnte nicht geladen werden.")
    st.code(str(error))
    st.stop()

class_names = load_class_names()

try:
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) != 4:
        st.error(f"Nicht unterstützte Eingabeform: {input_shape}")
        st.stop()

    _, image_height, image_width, channels = input_shape

    if image_height is None or image_width is None:
        st.error("Die Bildgröße konnte nicht erkannt werden.")
        st.stop()

    image_height = int(image_height)
    image_width = int(image_width)

except Exception as error:
    st.error(f"Fehler beim Ermitteln der Bildgröße: {error}")
    st.stop()

st.caption(
    f"Erwartete Bildgröße: {image_width} × {image_height} Pixel"
)

uploaded_file = st.file_uploader(
    "Bild hochladen",
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
            with st.spinner("Das Bild wird analysiert ..."):
                image_array = preprocess_image(
                    image,
                    image_width,
                    image_height
                )

                label, confidence = make_prediction(
                    model,
                    image_array,
                    class_names
                )

            st.success(f"Ergebnis: {label}")
            st.metric(
                "Sicherheit",
                f"{confidence * 100:.2f} %"
            )

    except Exception as error:
        st.error("Das Bild konnte nicht analysiert werden.")
        st.code(str(error))
