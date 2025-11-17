import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# configurar la página
st.set_page_config(
    page_title="Clasificador de Residos Orgánicos e Inorgánicos",
    layout="wide",
    page_icon="♻️",
    initial_sidebar_state="expanded"
)

if 'total_classifications' not in st.session_state:
    st.session_state.total_classifications = 0


# funciones aux
@st.cache_resource
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None


def get_waste_info(class_name):
    info = {
        "Orgánico": {
            "emoji": "🥬",
            "color": "#4CAF50"
        },
        "Inorgánico": {
            "emoji": "🥫",
            "color": "#FF6347"
        }
    }
    return info.get(class_name, {"emoji": "❓", "color": "#999999"})


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array /= 255.0
    return image_array


def predict_waste(model, image):
    prediction = model.predict(image, verbose=0)
    probability = float(prediction[0][0])

    if probability >= 0.5:
        class_name = "Inorgánico"
        confidence = probability
    else:
        class_name = "Orgánico"
        confidence = 1 - probability

    return class_name, confidence, probability


st.sidebar.title("Configuración")

st.sidebar.subheader("Modelo de IA")

modelo_info = {
    'MobileNetV2 (Rápido)': {
        'path': 'models/mobilenetv2_residuos.h5',
        'description': 'Modelo basado en MobileNetV2, optimizado para velocidad y eficiencia en dispositivos móviles.',
        'accuracy': '94%'
    },
    'AlexNet (Balanceado)': {
        'path': 'models/alexnet_residuos.h5',
        'description': 'Modelo basado en AlexNet, equilibrando precisión y velocidad para aplicaciones generales.',
        'accuracy': '92%'
    },
    'HRNet (Alta Resolución)': {
        'path': 'models/hrnet_residuos.h5',
        'description': 'Modelo basado en HRNet, mantiene alta resolución para máxima precisión en detalles espaciales.',
        'accuracy': '93%'
    }
}

modelo_seleccionado = st.sidebar.selectbox(
    "Selecciona el modelo de IA:",
    list(modelo_info.keys())
)

# header principal

st.title("♻️ Clasificador Inteligente de Residuos")
st.markdown("""
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='color: #1f77b4; margin-top: 0;'>🤖 Inteligencia Artificial para Gestión de Residuos</h3>
        <p style='font-size: 16px; color: #555;'>
            Sube una imagen de un residuo y nuestra IA lo clasificará en <b style='color: #4CAF50;'>Orgánico</b> 
            o <b style='color: #FF6347;'>Inorgánico</b> usando modelos de Deep Learning entrenados con miles de imágenes.
        </p>
    </div>
""", unsafe_allow_html=True)

# Sección de carga de imagen

col_upload, col_result = st.columns([1, 1])

with col_upload:
    st.subheader("Subir Imagen")

    uploaded_file = st.file_uploader("Elige una imagen de residuo (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagen Cargada', use_container_width=True)
        clasificar_button = st.button("Clasificar Imagen")

    else:
        st.info("Por favor, sube una imagen para clasificar.")
        clasificar_button = False

with col_result:
    st.subheader("Resultado de Clasificación")

    if uploaded_file is not None and clasificar_button:
        with st.spinner(f"Cargando modelo {modelo_seleccionado}..."):
            model_path = modelo_info[modelo_seleccionado]['path']

            if not os.path.exists(model_path):
                st.error(f"El modelo seleccionado no se encuentra en la ruta: {model_path}")
                st.stop()

            model = load_model(model_path)

        if model is None:
            st.error("No se pudo cargar el modelo. Verifica la ruta y el archivo del modelo.")
            st.stop()

        # Procesar imagen
        with st.spinner("Procesando imagen..."):
            processed_image = preprocess_image(image, target_size=(224, 224))

        # Predecir
        with st.spinner("Clasificando..."):
            clase, confianza, probabilidad = predict_waste(model, processed_image)

        st.session_state.total_classifications += 1

    # Mostrar resultados
        info = get_waste_info(clase)
        st.markdown(f"""
            <div style='background-color: {info['color']}; padding: 30px; border-radius: 15px; text-align: center;'>
            <h1 style='color: white; margin: 0; font-size: 48px;'>{info['emoji']}</h1>
            <h2 style='color: white; margin: 10px 0;'>{clase}</h2>
            <p style='color: white; font-size: 24px; margin: 0;'>{confianza * 100:.1f}% de confianza</p>
        </div>
        """, unsafe_allow_html=True)

        st.success(f"Clasificación completada con {modelo_seleccionado}")

# footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><b>♻️ Clasificador Inteligente de Residuos</b></p>
        <p>Desarrollado con TensorFlow, Keras y Streamlit</p>
        <p style='font-size: 12px;'>
            🌱 Ayudando a crear un mundo más sostenible a través de la tecnología 🌍
        </p>
    </div>
""", unsafe_allow_html=True)