# src/train_model.py
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "stress_model.pkl")


def train_model():
    load_dotenv()

    DATABASE_URL_RAW = os.getenv("DATABASE_URL")

    if not DATABASE_URL_RAW:
        raise ValueError("No se encontró DATABASE_URL en el archivo .env o variables del entorno")

    DATABASE_URL = DATABASE_URL_RAW.split("?")[0]

    print("Conectando a PostgreSQL...", flush=True)
    engine = create_engine(DATABASE_URL)

    query = """
    SELECT "heartRateAvg", "sleepHours", "steps", "screenTimeMinutes", 
           "socialMediaMin", "moodScore", "perceivedStress" 
    FROM "DailyMetric";
    """

    print("Descargando datos de entrenamiento...", flush=True)
    df = pd.read_sql(query, engine)

    print(f"Se han extraído {len(df)} registros de la base de datos.", flush=True)

    if df.empty:
        raise ValueError("No hay datos en la tabla DailyMetric para entrenar el modelo")

    def calculate_stress(row):
        if row["sleepHours"] < 5 or row["perceivedStress"] >= 8 or row["heartRateAvg"] > 85:
            return 2  # HIGH
        elif row["sleepHours"] < 6.5 or row["perceivedStress"] >= 5:
            return 1  # MEDIUM
        return 0  # LOW

    df["stress_label"] = df.apply(calculate_stress, axis=1)

    X = df.drop("stress_label", axis=1)
    y = df["stress_label"]

    print("Entrenando el modelo Random Forest...", flush=True)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)

    print(f"Modelo entrenado con éxito y guardado en: {MODEL_PATH}", flush=True)

    return {
        "message": "Modelo entrenado con éxito",
        "records_used": int(len(df)),
        "model_path": MODEL_PATH
    }