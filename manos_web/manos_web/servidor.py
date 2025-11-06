from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

# === Cargar el modelo entrenado ===
modelo = tf.keras.models.load_model("modelo_letras.h5")
print("✅ Modelo cargado correctamente")

# Si tienes datos.pkl lo cargamos, pero sin asumir que tiene media/std
try:
    with open("datos.pkl", "rb") as f:
        datos = pickle.load(f)
    print("✅ datos.pkl cargado correctamente")
except Exception as e:
    datos = None
    print("⚠️ No se cargó datos.pkl:", e)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        landmarks = np.array(data['landmarks']).reshape(1, -1)

        # Normalizar (si tienes media/std podrías usarlos)
        landmarks = (landmarks - 0.5) / 0.2

        prediction = modelo.predict(landmarks)
        letra_idx = np.argmax(prediction)
        letras = list("ABCDEFGHIJKLMNÑOPQRSTUVWXYZ")
        letra = letras[letra_idx] if letra_idx < len(letras) else '?'

        return jsonify({'letra': letra})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("🚀 Servidor iniciado en http://127.0.0.1:5000")
    app.run(debug=True)
