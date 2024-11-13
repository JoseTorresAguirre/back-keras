from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np

# Cargar el modelo
model = load_model('model/final.keras')

app = Flask(__name__)
# Habilitar CORS para todas las rutasp
CORS(app)

#USAR METODO POST
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del formulario (en JSON)
    data = request.get_json()
    user_input = np.array(list(data['input'].values())).reshape(1, -1)  # Convertir el diccionario a un array 2D
    
    # Predecir
    prediction = model.predict(user_input)
    prediction_class = 1 if prediction > 0.5 else 0  # Usamos un umbral de 0.5 para determinar si es hora punta
    
    # Enviar respuesta
    return jsonify({'prediction': int(prediction_class)})

# Ruta GET para obtener información sobre el estado del servidor o el modelo
@app.route('/predict', methods=['GET'])
def get_status():
    # Aquí puedes retornar información del modelo o el estado del servidor
    return jsonify({'message': 'Servidor funcionando, listo para recibir solicitudes POST.'})


if __name__ == '__main__':
    app.run(debug=True)
