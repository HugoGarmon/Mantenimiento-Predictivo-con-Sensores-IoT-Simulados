import uvicorn
from fastapi import FastAPI
import tensorflow as tf 
import json
import numpy as np
from influxdb_client import InfluxDBClient
import os
from datetime import datetime
import xgboost as xgb

app = FastAPI(title="Industrial AI API - Semana 3 Final")

# --- 1. CARGA DE MODELOS ---
MODEL_PATH = "models/"
try:
    # Modelos Keras (LSTM y Autoencoder)
    lstm_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'lstm_model.keras'))
    autoencoder = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'autoencoder_model.keras'))
    # Modelo XGBoost desde JSON
    xgboost_model = xgb.Booster()
    xgboost_model.load_model(os.path.join(MODEL_PATH, 'xgboost_model.json'))
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
        # Traer las últimas 30 muestras y pivotarlas para obtener columnas por campo
        query = f'''
        from(bucket: "{BUCKET}")
            |> range(start: -30m)
            |> filter(fn: (r) => r["_measurement"] == "telemetria_maquinaria")
            |> filter(fn: (r) => r["_field"] == "temperatura" or r["_field"] == "vibracion" or r["_field"] == "presion" or r["_field"] == "rpm")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns:["_time"])
            |> tail(n:30)
        '''
        result = query_api.query(org=ORG, query=query)

        # Construir lista de puntos ordenada (antiguo -> reciente)
        points = []
        for table in result:
            for record in table.records:
                vals = record.values
                points.append({
                    "timestamp": vals.get("_time"),
                    "temperatura": vals.get("temperatura"),
                    "vibracion": vals.get("vibracion"),
                    "presion": vals.get("presion"),
                    "rpm": vals.get("rpm")
                })

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
    """Dado un listado de hasta 30 puntos (orden ascendente) construye
    una matriz (30, 22) con features originales + features sintéticas.
    Si hay menos de 30 puntos se hace padding replicando el último valor.
    """
    # Normalizar/convertir y garantizar 30 pasos
    max_len = 30
    pts = list(points)
    if len(pts) < max_len:
        if len(pts) == 0:
            # Rellenar con ceros
            pts = [{"temperatura": 0.0, "vibracion": 0.0, "presion": 0.0, "rpm": 0.0}] * max_len
        else:
            last = pts[-1]
            while len(pts) < max_len:
                pts.append(last)

    # Extraer arrays por sensor (valores crudos)
    temps_raw = [float(p.get("temperatura") or 0.0) for p in pts]
    vibs_raw = [float(p.get("vibracion") or 0.0) for p in pts]
    press_raw = [float(p.get("presion") or 0.0) for p in pts]
    rpms_raw = [float(p.get("rpm") or 0.0) for p in pts]

    # Normalizar como en el preprocesado de entrenamiento
    temps = [t / 100.0 for t in temps_raw]
    vibs = [v / 10.0 for v in vibs_raw]
    press = [p / 50.0 for p in press_raw]
    rpms = [r / 3000.0 for r in rpms_raw]

    features = []
    for i in range(max_len):
        # valores actuales (normalizados)
        t = temps[i]
        v = vibs[i]
        pr = press[i]
        r = rpms[i]

        # prev window para cálculos (incluye i)
        start = max(0, i-2)
        window_t = temps[start:i+1]
        window_v = vibs[start:i+1]
        window_pr = press[start:i+1]
        window_r = rpms[start:i+1]

        # diffs (normalizados)
        t_diff = t - temps[i-1] if i > 0 else 0.0
        v_diff = v - vibs[i-1] if i > 0 else 0.0
        pr_diff = pr - press[i-1] if i > 0 else 0.0
        r_diff = r - rpms[i-1] if i > 0 else 0.0

        # medias móviles (3) sobre valores normalizados
        t_ma3 = float(sum(window_t) / len(window_t))
        v_ma3 = float(sum(window_v) / len(window_v))
        pr_ma3 = float(sum(window_pr) / len(window_pr))
        r_ma3 = float(sum(window_r) / len(window_r))

        # cuadrados (sobre valores normalizados)
        t_sq = t * t
        v_sq = v * v
        pr_sq = pr * pr
        r_sq = r * r

        # RMS (sobre ventana, normalizado)
        t_rms = (sum([x*x for x in window_t]) / len(window_t)) ** 0.5
        v_rms = (sum([x*x for x in window_v]) / len(window_v)) ** 0.5
        pr_rms = (sum([x*x for x in window_pr]) / len(window_pr)) ** 0.5
        r_rms = (sum([x*x for x in window_r]) / len(window_r)) ** 0.5

        # rango en ventana (max-min) sobre valores normalizados
        t_range = max(window_t) - min(window_t)
        v_range = max(window_v) - min(window_v)

        row = [
            t, v, pr, r,
            t_diff, v_diff, pr_diff, r_diff,
            t_ma3, v_ma3, pr_ma3, r_ma3,
            t_sq, v_sq, pr_sq, r_sq,
            t_rms, v_rms, pr_rms, r_rms,
            t_range, v_range
        ]
        features.append(row)

    return np.array(features, dtype=np.float32)


def determine_cause(points):
    """Determina el parámetro más responsable de la anomalía comparando
    el último valor con la mediana histórica (excluyendo el último).
    Devuelve una cadena: 'Temperatura', 'Vibración', 'Presión', 'RPM' o None.
    """
    if not points or len(points) < 2:
        return None

    # Extraer arrays crudos
    temps = [float(p.get("temperatura") or 0.0) for p in points]
    vibs = [float(p.get("vibracion") or 0.0) for p in points]
    press = [float(p.get("presion") or 0.0) for p in points]
    rpms = [float(p.get("rpm") or 0.0) for p in points]

    # Último valor y mediana histórica (excluyendo último)
    last_idx = len(points) - 1
    last_vals = {
        'Temperatura': temps[last_idx],
        'Vibración': vibs[last_idx],
        'Presión': press[last_idx],
        'RPM': rpms[last_idx]
    }

    import statistics
    med_vals = {
        'Temperatura': statistics.median(temps[:-1]),
        'Vibración': statistics.median(vibs[:-1]),
        'Presión': statistics.median(press[:-1]),
        'RPM': statistics.median(rpms[:-1])
    }

    # Normalizar diferencias según escalas de entrenamiento
    scales = {'Temperatura': 100.0, 'Vibración': 10.0, 'Presión': 50.0, 'RPM': 3000.0}

    diffs = {}
    for k in last_vals:
        try:
            diff = (last_vals[k] - med_vals[k]) / scales[k]
        except Exception:
            diff = 0.0
        diffs[k] = abs(diff)

    # Seleccionar máxima diferencia
    cause = max(diffs, key=diffs.get)

    # Si el mayor desplazamiento es muy pequeño, devolver None
    if diffs[cause] < 0.02:  # umbral: 2% normalizado
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
            # Construir matriz de features (30,22) usando datos reales
            feature_matrix = build_feature_matrix(data)
            # Para la respuesta, devolver el último punto como sensor_data
            last_point = data[-1]
        else:
            # Compatibilidad: si data es un dict con solo la última muestra
            last_point = data
            # Normalización manual (fallback)
            temp_n = float(data.get('temperatura', 0)) / 100.0
            vib_n = float(data.get('vibracion', 0)) / 10.0
            pres_n = float(data.get('presion', 0)) / 50.0
            rpm_n = float(data.get('rpm', 0)) / 3000.0
            features_22 = [temp_n, vib_n, pres_n, rpm_n] + [0.0] * 18
            feature_matrix = np.array([features_22] * 30).astype(np.float32)

        input_keras = np.expand_dims(feature_matrix, axis=0)  # Shape: (1, 30, 22)

        # Aplanado para XGBoost (30 * 22 = 660 características)
        input_xgb = feature_matrix.flatten().reshape(1, -1)  # Shape: (1, 660)

        # C. INFERENCIA REAL
        # 1. Autoencoder (Salud)
        reconstructed = autoencoder.predict(input_keras, verbose=0)
        mse = np.mean(np.square(input_keras - reconstructed))

        # 2. LSTM (RUL)
        rul_val = float(lstm_model.predict(input_keras, verbose=0)[0][0])

        # 3. XGBoost (Tipo de fallo) - Requiere DMatrix
        dmatrix_input = xgb.DMatrix(input_xgb)
        fail_code = int(xgboost_model.predict(dmatrix_input)[0])

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
                "is_anomaly": bool(es_anomalo), # Ahora depende de la salud real
                "rul_estimated": round(float(abs(rul_val)), 1),
                "status": "ESTABLE" if not es_anomalo else "ANOMALÍA",
                "failure_code": int(fail_code),
                "cause": cause
            }
        }

    except Exception as e:
        print(f"❌ Error en la inferencia IA: {e}")
        return {"status": "error", "message": str(e)}

# --- 4. ARRANQUE ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)