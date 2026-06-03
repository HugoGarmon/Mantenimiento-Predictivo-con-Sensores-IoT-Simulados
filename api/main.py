import uvicorn
from fastapi import FastAPI
import tensorflow as tf 
import json
import numpy as np
import pandas as pd
import joblib
from influxdb_client import InfluxDBClient
import os
from datetime import datetime
import xgboost as xgb

app = FastAPI(title="Industrial AI API - Semana 3 Final")

# --- 1. CARGA DE MODELOS ---
MODEL_PATH = "models/"
try:
    # Modelos Keras (LSTM y Autoencoder)
    lstm_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'lstm_regresion_rul.keras'))
    autoencoder = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'autoencoder_anomalias.keras'))
    # Modelo XGBoost desde JSON
    xgboost_model = xgb.Booster()
    xgboost_model.load_model(os.path.join(MODEL_PATH, 'xgboost_clasificacion.json'))
    # Scaler
    scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
    print("[SISTEMA] Todos los modelos y el scaler fueron cargados exitosamente.")
except Exception as e:
    print(f"[ERROR] Fallo al cargar modelos o scaler: {e}")

# --- 2. CONFIGURACIÓN INFLUXDB ---
URL = "http://localhost:8086"
TOKEN = "6wtWMbUQhEJfbEGL-JiVWpF-rL0jidkZnAkvrR1hSaPRTDKeh7zP-ep0NWyeQ3EOzKVgvctIAj8aLas1NQqXYQ=="
ORG = "docs"
BUCKET = "home"

# Constantes de columnas de entrenamiento
SENSORES_CONSTANTES = ['ajuste_3', 'sensor_1', 'sensor_10', 'sensor_18', 'sensor_19']
SENSORES_REDUNDANTES = ['sensor_14']
SENSORES_PARA_FEATURES = ['sensor_4', 'sensor_11', 'sensor_15', 'sensor_21']

def get_real_data():
    """Obtiene las últimas 30 muestras de telemetría desde InfluxDB."""
    try:
        client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
        query_api = client.query_api()
        # Traer las últimas 30 muestras de todos los sensores y ajustes
        query = f'''
        from(bucket: "{BUCKET}")
            |> range(start: -30m)
            |> filter(fn: (r) => r["_measurement"] == "telemetria_maquinaria")
            |> filter(fn: (r) => r["_field"] =~ /^(sensor_|ajuste_|ciclo)/)
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns:["_time"])
            |> tail(n:30)
        '''
        result = query_api.query(org=ORG, query=query)

        points = []
        for table in result:
            for record in table.records:
                vals = record.values
                row = {
                    "timestamp": vals.get("_time"),
                    "id_motor": int(vals.get("id_motor") or 1),
                    "ciclo": int(vals.get("ciclo") or 0)
                }
                for i in range(1, 4):
                    row[f"ajuste_{i}"] = float(vals.get(f"ajuste_{i}") or 0.0)
                for i in range(1, 22):
                    row[f"sensor_{i}"] = float(vals.get(f"sensor_{i}") or 0.0)
                points.append(row)

        client.close()

        if not points:
            return None

        # Asegurar orden ascendente por tiempo
        points = sorted(points, key=lambda x: x["timestamp"])
        return points
    except Exception as e:
        print(f"⚠️ Error InfluxDB: {e}")
        return None


def build_feature_matrix(points):
    """Dado un listado de puntos (orden ascendente), construye
    la matriz de características preprocesada y normalizada de shape (30, 24).
    """
    df = pd.DataFrame(points)
    
    # Eliminar posibles duplicados de timestamp por múltiples tablas de InfluxDB y ordenar
    if "timestamp" in df.columns:
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    
    if len(df) > 30:
        df = df.tail(30).reset_index(drop=True)
    elif len(df) < 30:
        if len(df) == 0:
            # Crear una fila vacía con valores por defecto
            default_row = {"ciclo": 0, "id_motor": 1}
            for i in range(1, 4):
                default_row[f"ajuste_{i}"] = 0.0
            for i in range(1, 22):
                default_row[f"sensor_{i}"] = 0.0
            df = pd.DataFrame([default_row] * 30)
        else:
            last_row = df.iloc[-1]
            rows_to_add = 30 - len(df)
            padding_df = pd.DataFrame([last_row] * rows_to_add)
            df = pd.concat([df, padding_df], ignore_index=True)
            
    # 1. Eliminar constantes
    df_limpio = df.drop(columns=[col for col in SENSORES_CONSTANTES if col in df.columns], errors='ignore')
    
    # 2. Media móvil de 10 ciclos
    columnas_sensores = [col for col in df_limpio.columns if 'sensor' in col]
    df_limpio[columnas_sensores] = (
        df_limpio[columnas_sensores]
        .rolling(window=10, min_periods=1)
        .mean()
    )
    
    # 3. Eliminar redundantes
    df_final = df_limpio.drop(columns=[col for col in SENSORES_REDUNDANTES if col in df_limpio.columns], errors='ignore')
    
    # 4. Ingeniería de características
    for s in SENSORES_PARA_FEATURES:
        df_final[f'{s}_std'] = df_final[s].rolling(window=10).std()
        df_final[f'{s}_diff'] = df_final[s].diff(periods=5)
        
    df_final.fillna(0, inplace=True)
    
    # 5. Ordenar columnas para el Scaler
    columnas_ajustes = ['ajuste_1', 'ajuste_2']
    columnas_sensores_activos = [
        'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6', 'sensor_7',
        'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_15',
        'sensor_16', 'sensor_17', 'sensor_20', 'sensor_21'
    ]
    columnas_engineered = [
        'sensor_4_std', 'sensor_4_diff', 'sensor_11_std', 'sensor_11_diff',
        'sensor_15_std', 'sensor_15_diff', 'sensor_21_std', 'sensor_21_diff'
    ]
    
    columnas_para_scaler = columnas_ajustes + columnas_sensores_activos + columnas_engineered
    
    # Normalizar usando el scaler cargado
    df_scaled = df_final.copy()
    df_scaled[columnas_para_scaler] = scaler.transform(df_scaled[columnas_para_scaler])
    
    # 6. Seleccionar 24 características de sensores
    columnas_sensores_finales = columnas_sensores_activos + columnas_engineered
    return df_scaled[columnas_sensores_finales].values.astype(np.float32)


def determine_cause(points):
    """Determina el sensor más responsable del fallo/anomalía comparando
    el último valor de los sensores críticos con su mediana histórica en la ventana.
    """
    if not points or len(points) < 2:
        return None
    
    # Evaluar los 4 sensores críticos
    sensores_eval = {
        'sensor_4': "Temperatura de Salida LPC (Sensor 4)",
        'sensor_11': "Presión Estática LPC (Sensor 11)",
        'sensor_15': "Relación de Derivación / Bypass (Sensor 15)",
        'sensor_21': "Purga de Refrigeración LPT (Sensor 21)"
    }
    
    import statistics
    diffs = {}
    for s, name in sensores_eval.items():
        vals = [float(p.get(s) or 0.0) for p in points]
        last_val = vals[-1]
        med_val = statistics.median(vals[:-1])
        
        divisor = abs(med_val) if abs(med_val) > 1e-5 else 1.0
        diffs[name] = abs(last_val - med_val) / divisor
        
    cause = max(diffs, key=diffs.get)
    
    if diffs[cause] < 0.01:
        return None
    return cause

# --- 3. ENDPOINT DE PREDICCIÓN ---
@app.get("/predict")
def predict():
    data = get_real_data()
    if not data:
        return {"status": "error", "message": "Esperando datos de InfluxDB..."}

    try:
        # Si get_real_data devolvió una lista de puntos (últimas 30 muestras), usarla
        if isinstance(data, list):
            feature_matrix = build_feature_matrix(data)
            last_point = data[-1]
        else:
            # Compatibilidad fallback
            last_point = data
            features_24 = [0.0] * 24
            feature_matrix = np.array([features_24] * 30).astype(np.float32)

        input_keras = np.expand_dims(feature_matrix, axis=0)  # Shape: (1, 30, 24)

        # Aplanado para XGBoost (30 * 24 = 720 características)
        input_xgb = feature_matrix.flatten().reshape(1, -1)  # Shape: (1, 720)

        # C. INFERENCIA REAL
        # 1. Autoencoder (Salud)
        reconstructed = autoencoder.predict(input_keras, verbose=0)
        mse = np.mean(np.square(input_keras - reconstructed))

        # 2. LSTM (RUL)
        rul_val = float(lstm_model.predict(input_keras, verbose=0)[0][0])

        # 3. XGBoost (Tipo de fallo / RUL <= 30) - Booster.predict devuelve probabilidades
        dmatrix_input = xgb.DMatrix(input_xgb)
        preds_xgb = xgboost_model.predict(dmatrix_input)
        prob_fail = preds_xgb[0]
        if hasattr(prob_fail, '__len__') or isinstance(prob_fail, np.ndarray):
            prob_fail = prob_fail[0]
        fail_code = 1 if float(prob_fail) > 0.5 else 0

        # D. CÁLCULO DE MÉTRICAS PARA INTERFAZ
        health_score = max(0.0, min(100.0, 100.0 - (float(mse) * 1000)))
        
        if health_score < 1:
            health_score = 100.0 - (float(np.log1p(mse)) * 20)
        
        # 2. NUEVA LÓGICA DE ESTADO: Basada en el Health Score final
        es_anomalo = health_score < 80.0 

        # Determinar causa probable a partir de la ventana de puntos
        cause = determine_cause(data) if isinstance(data, list) else None

        # Preparar sensor_data a partir del último punto
        sensor_data = {}
        if isinstance(last_point, dict):
            for k, v in last_point.items():
                try:
                    sensor_data[k] = float(v)
                except Exception:
                    sensor_data[k] = v
        else:
            sensor_data = last_point

        return {
            "sensor_data": sensor_data,
            "prediction": {
                "health_score": round(float(health_score), 2),
                "is_anomaly": bool(es_anomalo),
                "rul_estimated": round(float(abs(rul_val)), 1),
                "status": "ESTABLE" if not es_anomalo else "ANOMALÍA",
                "failure_code": int(fail_code),
                "cause": cause
            }
        }

    except Exception as e:
        print(f"[ERROR] Error en la inferencia IA: {e}")
        return {"status": "error", "message": str(e)}

# --- 4. ARRANQUE ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)