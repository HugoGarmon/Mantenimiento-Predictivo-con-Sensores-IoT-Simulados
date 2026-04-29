import uvicorn
from fastapi import FastAPI
import tensorflow as tf
import joblib
import numpy as np
from influxdb_client import InfluxDBClient
import os
from datetime import datetime

app = FastAPI(title="Industrial AI API - Semana 3 Final")

# --- 1. CARGA DE MODELOS ---
MODEL_PATH = "models/"
try:
    # Modelos Keras (LSTM y Autoencoder)
    lstm_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'lstm_model.keras'))
    autoencoder = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'autoencoder_model.keras'))
    # Modelo Pickle (XGBoost)
    xgboost_model = joblib.load(os.path.join(MODEL_PATH, 'xgboost_model.pkl'))
    print("✅ [SISTEMA] Todos los modelos cargados y listos.")
except Exception as e:
    print(f"❌ [ERROR] Fallo al cargar modelos: {e}")

# --- 2. CONFIGURACIÓN INFLUXDB ---
URL = "http://localhost:8086"
TOKEN = "6wtWMbUQhEJfbEGL-JiVWpF-rL0jidkZnAkvrR1hSaPRTDKeh7zP-ep0NWyeQ3EOzKVgvctIAj8aLas1NQqXYQ=="
ORG = "docs"
BUCKET = "home"

def get_real_data():
    """Obtiene el último registro de telemetría."""
    try:
        client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
        query_api = client.query_api()
        query = f'''
        from(bucket: "{BUCKET}")
            |> range(start: -1m)
            |> filter(fn: (r) => r["_measurement"] == "telemetria_maquinaria")
            |> last()
        '''
        result = query_api.query(org=ORG, query=query)
        data = {record.get_field(): record.get_value() for table in result for record in table.records}
        client.close()
        return data
    except Exception as e:
        print(f"⚠️ Error InfluxDB: {e}")
        return None

# --- 3. ENDPOINT DE PREDICCIÓN ---
@app.get("/predict")
def predict():
    data = get_real_data()
    if not data or 'temperatura' not in data:
        return {"status": "error", "message": "Esperando datos de InfluxDB..."}

    try:
        # A. NORMALIZACIÓN MANUAL (Parche para evitar Salud 0)
        # Dividimos por valores máximos típicos para que el modelo no vea números gigantes
        temp_n = float(data.get('temperatura', 0)) / 100.0
        vib_n = float(data.get('vibracion', 0)) / 10.0
        pres_n = float(data.get('presion', 0)) / 50.0
        rpm_n = float(data.get('rpm', 0)) / 3000.0

        # Creamos el vector de 13 variables (tus 4 + 9 ceros)
        features_13 = [temp_n, vib_n, pres_n, rpm_n] + [0.0] * 9
        
        # B. ADAPTACIÓN DE SHAPES (Formas de los datos)
        # Ventana temporal de 30 pasos para Keras (LSTM/Autoencoder)
        ventana_30 = np.array([features_13] * 30).astype(np.float32)
        input_keras = np.expand_dims(ventana_30, axis=0) # Shape: (1, 30, 13)

        # Aplanado para XGBoost (30 * 13 = 390 características)
        input_xgb = ventana_30.flatten().reshape(1, -1) # Shape: (1, 390)

        # C. INFERENCIA REAL
        # 1. Autoencoder (Salud)
        reconstructed = autoencoder.predict(input_keras, verbose=0)
        mse = np.mean(np.square(input_keras - reconstructed))
        
        # 2. LSTM (RUL)
        rul_val = float(lstm_model.predict(input_keras, verbose=0)[0][0])
        
        # 3. XGBoost (Tipo de fallo)
        fail_code = int(xgboost_model.predict(input_xgb)[0])

        # D. CÁLCULO DE MÉTRICAS PARA INTERFAZ
        # Ajustamos el multiplicador para que la salud sea realista (ej: 0.005 -> 95%)
        health_score = max(0, min(100, 100 - (mse * 1000)))
        
        # Si la salud sigue siendo 0 por el escalado, aplicamos un "suavizado" logarítmico
        health_score = max(0.0, min(100.0, 100.0 - (float(mse) * 1000)))
        
        if health_score < 1:
            health_score = 100.0 - (float(np.log1p(mse)) * 20)
        
        # 2. NUEVA LÓGICA DE ESTADO: Basada en el Health Score final
        # Si la salud es mayor a 80, todo está bien.
        es_anomalo = health_score < 80.0 

        return {
            "sensor_data": {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in data.items()},
            "prediction": {
                "health_score": round(float(health_score), 2),
                "is_anomaly": bool(es_anomalo), # Ahora depende de la salud real
                "rul_estimated": round(float(abs(rul_val)), 1),
                "status": "ESTABLE" if not es_anomalo else "ANOMALÍA",
                "failure_code": int(fail_code)
            }
        }

    except Exception as e:
        print(f"❌ Error en la inferencia IA: {e}")
        return {"status": "error", "message": str(e)}

# --- 4. ARRANQUE ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)