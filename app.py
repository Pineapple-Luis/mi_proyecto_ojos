from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# ============================================
# 1. CONFIGURACIÓN
# ============================================
TARGET_SIZE = (224, 224)
CLASS_NAMES = [
    'ojos_almendrados',
    'ojos_saltones',
    'ojos_juntos',
    'ojos_pequeños',
    'ojos_asiáticos',
    'ojos_caídos',
    'ojos_encapotados'
]

# ============================================
# 2. CARGAR EL MODELO (con verificación de archivo)
# ============================================
model = None
model_paths = ['modelo_final.h5', 'model.h5', 'modelo_ojos.h5']

for path in model_paths:
    if os.path.exists(path):
        print(f"🔍 Intentando cargar: {path}")
        try:
            model = tf.keras.models.load_model(path)
            print(f"✅ Modelo cargado exitosamente desde {path}")
            break
        except Exception as e:
            print(f"❌ Error cargando {path}: {e}")
    else:
        print(f"⚠️ Archivo no encontrado: {path}")

if model is None:
    print("❌ No se pudo cargar ningún modelo. Verifica que el archivo .h5 exista.")
else:
    print("✅ Modelo listo para usar.")

# ============================================
# 3. PREPROCESAMIENTO
# ============================================
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(TARGET_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"❌ Error en preprocesamiento: {e}")
        raise

# ============================================
# 4. RUTAS
# ============================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("📥 Solicitud recibida en /predict")
    
    # Verificar modelo
    if model is None:
        print("❌ Modelo no cargado")
        return jsonify({'error': 'Modelo no cargado'}), 500
    
    # Verificar archivo en la solicitud
    if 'image' not in request.files:
        print("❌ No se encontró 'image' en request.files")
        return jsonify({'error': 'No se envió ninguna imagen'}), 400
    
    file = request.files['image']
    if file.filename == '':
        print("❌ Nombre de archivo vacío")
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    
    try:
        # Leer imagen
        print("📸 Leyendo imagen...")
        image_bytes = file.read()
        
        # Preprocesar
        print("🔄 Preprocesando imagen...")
        processed_image = preprocess_image(image_bytes)
        
        # Predecir
        print("🧠 Haciendo predicción...")
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        predicted_class = CLASS_NAMES[predicted_index]
        
        # Top 3
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3 = [{'clase': CLASS_NAMES[i], 'probabilidad': float(predictions[0][i] * 100)} for i in top_3_indices]
        
        print(f"✅ Predicción exitosa: {predicted_class} con {confidence:.2f}%")
        
        return jsonify({
            'success': True,
            'clase': predicted_class,
            'confianza': round(confidence, 2),
            'top_3': top_3
        })
        
    except Exception as e:
        print(f"❌ Error en /predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ============================================
# 5. LANZAMIENTO
# ============================================
if __name__ == '__main__':
    from waitress import serve
    import os
    port = int(os.environ.get('PORT', 5000))
    serve(app, host='0.0.0.0', port=port)
