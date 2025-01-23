
import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Cargar el modelo
model = load_model('/content/drive/MyDrive/pet_vision_file/pet_vision_ai_model.keras')

# Crear la interfaz
st.title("Pet Vision")
st.write("Sube una imagen de una mascota para que la clasifiquemos como gato o perro.")

# Subir una imagen
img = st.file_uploader("Sube una imagen de una mascota", type=["jpg", "png", "jpeg"])

# Mensaje de referencia
st.markdown("**Creado por Onkar Dass Kaur**")

if img:
    # Mostrar la imagen subida
    st.image(img, caption="Imagen subida", use_column_width=True)

    # Preprocesar la imagen
    img = Image.open(img)
    img = img.resize((150, 150))  # Cambiar el tamaño a 150x150, que es el tamaño de entrada de tu modelo
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión extra para que sea compatible con el modelo

    # Normalizar la imagen
    img_array = img_array / 255.0

    # Hacer la predicción
    pred = model.predict(img_array)

    # Mostrar el resultado de la predicción con iconos
    if pred[0][0] > 0.5:
        st.write("¡Es un perro!")
        st.image("https://img.icons8.com/ios/452/dog.png", width=100)  # Icono de perro
    else:
        st.write("¡Es un gato!")
        st.image("https://img.icons8.com/ios/452/cat.png", width=100)  # Icono de gato

        # Mensaje de agradecimiento
    st.write("Gracias por usar nuestra herramienta Pet Vision.")

