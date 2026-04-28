# train_model.py
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from dotenv import load_dotenv

# 1. Cargar la URL de la base de datos desde el archivo .env
load_dotenv()
DATABASE_URL_RAW = os.getenv('DATABASE_URL')

if not DATABASE_URL_RAW:
    raise ValueError("No se encontró DATABASE_URL en el archivo .env")

# LÍNEA CORREGIDA: Limpiar la URL para quitar '?schema=public' que causa error en psycopg2
DATABASE_URL = DATABASE_URL_RAW.split('?')[0]

print("Conectando a PostgreSQL...")
engine = create_engine(DATABASE_URL)
# 2. Extraer los datos reales de la tabla DailyMetric
# Nota: Prisma crea las tablas con comillas dobles (Case Sensitive) por defecto en Postgres
query = """
SELECT "heartRateAvg", "sleepHours", "steps", "screenTimeMinutes", 
       "socialMediaMin", "moodScore", "perceivedStress" 
FROM "DailyMetric";
"""

print("Descargando datos de entrenamiento...")
df = pd.read_sql(query, engine)

print(f"✅ Se han extraído {len(df)} registros de la base de datos.")

# 3. Regla lógica para generar la etiqueta de estrés basada en los datos extraídos
# (0: LOW, 1: MEDIUM, 2: HIGH)
def calculate_stress(row):
    if row['sleepHours'] < 5 or row['perceivedStress'] >= 8 or row['heartRateAvg'] > 85:
        return 2 # HIGH
    elif row['sleepHours'] < 6.5 or row['perceivedStress'] >= 5:
        return 1 # MEDIUM
    return 0 # LOW

df['stress_label'] = df.apply(calculate_stress, axis=1)

# 4. Entrenar el modelo
X = df.drop('stress_label', axis=1)
y = df['stress_label']

print("Entrenando el modelo Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. Guardar el modelo
joblib.dump(model, 'stress_model.pkl')
print("✅ Modelo entrenado con éxito usando datos de PostgreSQL y guardado como 'stress_model.pkl'")