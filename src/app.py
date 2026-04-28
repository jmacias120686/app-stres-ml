# src/app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import traceback

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "stress_model.pkl")

model = None

STRESS_LEVELS = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH"
}


def load_model():
    global model

    if model is None:
        print("Cargando modelo desde:", MODEL_PATH)
        print("Existe el modelo:", os.path.exists(MODEL_PATH))

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No se encontró stress_model.pkl en: {MODEL_PATH}")

        model = joblib.load(MODEL_PATH)

    return model


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "API ML funcionando correctamente",
        "status": "ok"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "stress-ml-api"
    })


@app.route("/predict", methods=["POST"])
def predict_stress():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Debe enviar un JSON válido"
            }), 400

        required_fields = [
            "heartRateAvg",
            "sleepHours",
            "steps",
            "screenTimeMinutes",
            "socialMediaMin",
            "moodScore",
            "perceivedStress"
        ]

        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return jsonify({
                "error": "Faltan campos obligatorios",
                "missingFields": missing_fields
            }), 400

        features = pd.DataFrame([{
            "heartRateAvg": data["heartRateAvg"],
            "sleepHours": data["sleepHours"],
            "steps": data["steps"],
            "screenTimeMinutes": data["screenTimeMinutes"],
            "socialMediaMin": data["socialMediaMin"],
            "moodScore": data["moodScore"],
            "perceivedStress": data["perceivedStress"]
        }])

        ml_model = load_model()

        prediction = ml_model.predict(features)[0]
        probabilities = ml_model.predict_proba(features)[0]
        max_prob = max(probabilities)

        trigger = "Ninguno"

        if data["sleepHours"] < 5:
            trigger = "Privación de sueño"
        elif data["perceivedStress"] >= 8:
            trigger = "Alta carga subjetiva"

        return jsonify({
            "level": STRESS_LEVELS[int(prediction)],
            "probability": float(max_prob),
            "triggerFactor": trigger
        })

    except Exception as e:
        print("Error en /predict:")
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)