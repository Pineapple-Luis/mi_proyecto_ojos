from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# 1. CONFIGURACIÓN (¡CÁMBIALO SEGÚN TU ENTRENAMIENTO!)
TARGET_SIZE = (224, 224)  # <--- AJUSTA AQUÍ (Ej: 128,128 o 299,299 si usaste Inception)
CLASS_NAMES = [
    'ojos_almendrados',
    'ojos_saltones',
    'ojos_juntos',
    'ojos_pequeños',
    'ojos_asiáticos',
    'ojos_caídos',
    'ojos_encapotados'
]  # <--- ASEGÚRATE QUE ESTE ORDEN COINCIDA CON EL DE TU entrenamiento

# 2. CARGAR EL MODELO (asegúrate que el archivo .h5 o .keras esté en la misma carpeta)
try:
    model = tf.keras.models.load_model('model.h5')  # Si es .keras, cámbialo
    print("✅ Modelo cargado exitosamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    model = None

def preprocess_image(image_bytes):
    """Convierte los bytes de la imagen a un array listo para el modelo"""
    img = Image.open(io.BytesIO(image_bytes))
    # Convertir a RGB por si acaso viene en PNG con transparencia
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Redimensionar
    img = img.resize(TARGET_SIZE)
    # Convertir a array y normalizar (si entrenaste con rescale=1./255, hazlo aquí)
    img_array = np.array(img) / 255.0  # Si NO normalizaste en entrenamiento, quita el /255.0
    # Agregar dimensión de batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no cargado'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    
    try:
        # Leer la imagen
        image_bytes = file.read()
        # Preprocesar
        processed_image = preprocess_image(image_bytes)
        # Inferencia
        predictions = model.predict(processed_image)
        # Obtener la clase con mayor probabilidad
        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        predicted_class = CLASS_NAMES[predicted_index]
        
        # (Opcional) Obtener top 3 para mostrarlos
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3 = [{'clase': CLASS_NAMES[i], 'probabilidad': float(predictions[0][i] * 100)} for i in top_3_indices]
        
        return jsonify({
            'success': True,
            'clase': predicted_class,
            'confianza': round(confidence, 2),
            'top_3': top_3
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5500)