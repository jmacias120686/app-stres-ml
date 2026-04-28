# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo previamente entrenado
model = joblib.load('stress_model.pkl')

# Mapeo de resultados
STRESS_LEVELS = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}

@app.route('/predict', methods=['POST'])
def predict_stress():
    try:
        data = request.json
        
        # Convertir los datos JSON a un DataFrame (debe coincidir con las columnas del entrenamiento)
        features = pd.DataFrame([{
            'heartRateAvg': data['heartRateAvg'],
            'sleepHours': data['sleepHours'],
            'steps': data['steps'],
            'screenTimeMinutes': data['screenTimeMinutes'],
            'socialMediaMin': data['socialMediaMin'],
            'moodScore': data['moodScore'],
            'perceivedStress': data['perceivedStress']
        }])
        
        # Predecir la clase y la probabilidad
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        max_prob = max(probabilities)
        
        # Determinar el trigger factor básico
        trigger = "Ninguno"
        if data['sleepHours'] < 5:
            trigger = "Privación de sueño"
        elif data['perceivedStress'] >= 8:
            trigger = "Alta carga subjetiva"
            
        return jsonify({
            'level': STRESS_LEVELS[prediction],
            'probability': float(max_prob),
            'triggerFactor': trigger
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # La API de Python correrá en el puerto 5000
    app.run(port=5000, debug=True)