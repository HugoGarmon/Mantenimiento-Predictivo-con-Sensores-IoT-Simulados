import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf 
import numpy as np
import pandas as pd
import joblib
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import os
from datetime import datetime
import xgboost as xgb
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import statistics
import logging

load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Industrial AI API - v1", version="1.0.0")

# --- CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api/v1")

# --- 1. CARGA DE MODELOS ---
MODEL_PATH = "models/"
lstm_model = None
autoencoder = None
xgboost_model = None
scaler = None

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
URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
TOKEN = os.getenv("INFLUXDB_TOKEN")
ORG = os.getenv("INFLUXDB_ORG", "docs")
BUCKET = os.getenv("INFLUXDB_BUCKET", "home")

# Cliente global Singleton
influx_client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
query_api = influx_client.query_api()
write_api = influx_client.write_api(write_options=SYNCHRONOUS)

@app.on_event("shutdown")
def shutdown_event():
    influx_client.close()

# Constantes de columnas de entrenamiento
SENSORES_CONSTANTES = ['ajuste_3', 'sensor_1', 'sensor_10', 'sensor_18', 'sensor_19']
SENSORES_REDUNDANTES = ['sensor_14']
SENSORES_PARA_FEATURES = ['sensor_4', 'sensor_11', 'sensor_15', 'sensor_21']

# --- PANTILLAS PYDANTIC PARA VALIDACIÓN Y DOCS ---
class PredictionDetail(BaseModel):
    health_score: float = Field(..., description="Score de salud unificado de 0 a 100%")
    is_anomaly: bool = Field(..., description="True si hay anomalía activa")
    rul_estimated: float = Field(..., description="Vida útil restante estimada en horas/ciclos")
    status: str = Field(..., description="Estado simplificado: ESTABLE o ANOMALÍA")
    severity_level: str = Field(..., description="Nivel de severidad de la alerta")
    recommended_action: str = Field(..., description="Acción recomendada de mantenimiento")
    failure_code: int = Field(..., description="Código de fallo según XGBoost (1 si RUL <= 30, 0 si no)")
    cause: Optional[str] = Field(None, description="Sensor causante de la desviación")

class PredictionResponse(BaseModel):
    sensor_data: Dict[str, Any] = Field(..., description="Últimos valores de sensores ingestados")
    prediction: PredictionDetail = Field(..., description="Predicción detallada")

class PredictionHistoryPoint(BaseModel):
    timestamp: str
    health_score: float
    rul_estimated: float
    is_anomaly: bool
    severity_level: str
    failure_code: int
    cause: Optional[str] = None

class PredictionHistoryResponse(BaseModel):
    id_motor: int
    history: List[PredictionHistoryPoint]

class MotorFleetInfo(BaseModel):
    id_motor: int
    health_score: float
    rul_estimated: float
    severity_level: str
    last_updated: str

class FleetStatusResponse(BaseModel):
    fleet_size: int
    severity_counts: Dict[str, int]
    motors: List[MotorFleetInfo]

class ModelMetadataResponse(BaseModel):
    models_loaded: Dict[str, bool]
    input_shape_keras: List[int]
    input_shape_xgb: List[int]
    scaler_columns: List[str]

class BatchPredictionRequest(BaseModel):
    motor_ids: List[int]

# --- 3. LÓGICA DE ALERTA Y SEVERIDAD ---
def get_severity_and_action(health_score: float) -> tuple[str, str]:
    if health_score >= 90.0:
        return "ÓPTIMO", "Operación normal. Continuar monitorizando."
    elif health_score >= 70.0:
        return "ATENCIÓN", "Desviación menor detectada. Programar inspección visual rutinaria en el próximo cambio de turno."
    elif health_score >= 50.0:
        return "ALERTA", "Degradación activa detectada. Programar mantenimiento correctivo no urgente dentro de las próximas 48 horas."
    elif health_score >= 30.0:
        return "CRÍTICO", "Alto riesgo de fallo inminente. Solicitar parada técnica programada inmediata."
    else:
        return "FALLO INMINENTE", "Fallo mecánico crítico detectado. Detener el motor de forma segura de inmediato."

# --- 4. FUNCIONES AUXILIARES DE INFLUXDB ---
def get_real_data(motor_id: int):
    """Obtiene las últimas 30 muestras de telemetría de un motor específico desde InfluxDB."""
    try:
        # Usar cliente global y consulta parametrizada
        query = '''
        from(bucket: bucket)
            |> range(start: -30m)
            |> filter(fn: (r) => r["_measurement"] == "telemetria_maquinaria")
            |> filter(fn: (r) => r["id_motor"] == motor_id)
            |> filter(fn: (r) => r["_field"] =~ /^(sensor_|ajuste_|ciclo)/)
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns:["_time"])
            |> tail(n:30)
        '''
        params = {"bucket": BUCKET, "motor_id": str(motor_id)}
        result = query_api.query(org=ORG, query=query, params=params)

        points = []
        for table in result:
            for record in table.records:
                vals = record.values
                row = {
                    "timestamp": vals.get("_time"),
                    "id_motor": int(vals.get("id_motor") or motor_id),
                    "ciclo": int(vals.get("ciclo") or 0)
                }
                for i in range(1, 4):
                    row[f"ajuste_{i}"] = float(vals.get(f"ajuste_{i}") or 0.0)
                for i in range(1, 22):
                    row[f"sensor_{i}"] = float(vals.get(f"sensor_{i}") or 0.0)
                points.append(row)

        if not points:
            return None

        # Asegurar orden ascendente por tiempo
        points = sorted(points, key=lambda x: x["timestamp"])
        return points
    except Exception as e:
        logger.error(f"⚠️ Error InfluxDB al leer telemetría (motor {motor_id}): {e}")
        return None

def save_prediction_to_db(motor_id: int, pred: PredictionDetail):
    """Guarda una predicción en InfluxDB en el measurement 'predicciones_ia'."""
    try:
        point = Point("predicciones_ia") \
            .tag("id_motor", str(motor_id)) \
            .field("health_score", float(pred.health_score)) \
            .field("rul_estimated", float(pred.rul_estimated)) \
            .field("is_anomaly", bool(pred.is_anomaly)) \
            .field("severity_level", str(pred.severity_level)) \
            .field("failure_code", int(pred.failure_code)) \
            .field("cause", str(pred.cause or ""))
        write_api.write(bucket=BUCKET, org=ORG, record=point)
    except Exception as e:
        logger.error(f"⚠️ Error InfluxDB al guardar predicción para motor {motor_id}: {e}")

def get_latest_prediction_from_db(motor_id: int) -> Optional[dict]:
    """Obtiene la última predicción guardada para un motor."""
    try:
        query = '''
        from(bucket: bucket)
            |> range(start: -24h)
            |> filter(fn: (r) => r["_measurement"] == "predicciones_ia")
            |> filter(fn: (r) => r["id_motor"] == motor_id)
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns:["_time"])
            |> tail(n:1)
        '''
        params = {"bucket": BUCKET, "motor_id": str(motor_id)}
        result = query_api.query(org=ORG, query=query, params=params)
        
        for table in result:
            for record in table.records:
                vals = record.values
                t = vals.get("_time")
                t_str = t.strftime("%Y-%m-%dT%H:%M:%SZ") if hasattr(t, "strftime") else str(t)
                return {
                    "health_score": float(vals.get("health_score") or 0.0),
                    "rul_estimated": float(vals.get("rul_estimated") or 0.0),
                    "severity_level": str(vals.get("severity_level") or "ÓPTIMO"),
                    "timestamp": t_str
                }
        return None
    except Exception as e:
        logger.error(f"⚠️ Error al recuperar predicción de DB (motor {motor_id}): {e}")
        return None

# --- 5. MATRIZ DE CARACTERÍSTICAS Y CAUSA PROBABLE ---
def build_feature_matrix(points):
    """Construye la matriz de características preprocesada de shape (30, 24)."""
    df = pd.DataFrame(points)
    
    if "timestamp" in df.columns:
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    
    if len(df) > 30:
        df = df.tail(30).reset_index(drop=True)
    elif len(df) < 30:
        if len(df) == 0:
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
            
    df_limpio = df.drop(columns=[col for col in SENSORES_CONSTANTES if col in df.columns], errors='ignore')
    
    columnas_sensores = [col for col in df_limpio.columns if 'sensor' in col]
    df_limpio[columnas_sensores] = (
        df_limpio[columnas_sensores]
        .rolling(window=5, min_periods=1)
        .mean()
    )
    
    df_final = df_limpio.drop(columns=[col for col in SENSORES_REDUNDANTES if col in df_limpio.columns], errors='ignore')
    
    for s in SENSORES_PARA_FEATURES:
        df_final[f'{s}_std'] = df_final[s].rolling(window=10).std()
        df_final[f'{s}_diff'] = df_final[s].diff(periods=5)
        
    df_final.fillna(0, inplace=True)
    
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
    
    df_scaled = df_final.copy()
    if scaler is not None:
        df_scaled[columnas_para_scaler] = scaler.transform(df_scaled[columnas_para_scaler])
    
    columnas_sensores_finales = columnas_sensores_activos + columnas_engineered
    return df_scaled[columnas_sensores_finales].values.astype(np.float32)

def determine_cause(points):
    """Determina el sensor más responsable del fallo/anomalía."""
    if not points or len(points) < 2:
        return None
    
    sensores_eval = {
        'sensor_4': "Temperatura de Salida LPC (Sensor 4)",
        'sensor_11': "Presión Estática LPC (Sensor 11)",
        'sensor_15': "Relación de Derivación / Bypass (Sensor 15)",
        'sensor_21': "Purga de Refrigeración LPT (Sensor 21)"
    }
    
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

# --- 6. PROCESAMIENTO DE INFERENCIA ---
def run_inference_for_motor(motor_id: int, data: list) -> PredictionDetail:
    """Ejecuta los modelos y guarda la predicción resultante."""
    feature_matrix = build_feature_matrix(data)
    
    input_keras = np.expand_dims(feature_matrix, axis=0)  # Shape: (1, 30, 24)
    input_xgb = feature_matrix.flatten().reshape(1, -1)   # Shape: (1, 720)
    
    # 1. Autoencoder (MSE)
    if autoencoder is not None:
        reconstructed = autoencoder.predict(input_keras, verbose=0)
        mse = np.mean(np.square(input_keras - reconstructed))
    else:
        mse = 0.02
        
    # 2. LSTM (RUL)
    if lstm_model is not None:
        rul_val = float(lstm_model.predict(input_keras, verbose=0)[0][0])
    else:
        rul_val = 130.0
        
    # 3. XGBoost (Clasificación de Fallo)
    if xgboost_model is not None:
        dmatrix_input = xgb.DMatrix(input_xgb)
        preds_xgb = xgboost_model.predict(dmatrix_input)
        prob_fail = preds_xgb[0]
        if hasattr(prob_fail, '__len__') or isinstance(prob_fail, np.ndarray):
            prob_fail = prob_fail[0]
        fail_code = 1 if float(prob_fail) > 0.5 else 0
    else:
        fail_code = 0
        
    health_score = max(0.0, min(100.0, 100.0 / (1.0 + np.exp((float(mse) - 0.05) * 80))))
    es_anomalo = health_score < 80.0
    
    severity_level, recommended_action = get_severity_and_action(health_score)
    cause = determine_cause(data)
    
    pred_detail = PredictionDetail(
        health_score=round(float(health_score), 2),
        is_anomaly=bool(es_anomalo),
        rul_estimated=round(float(abs(rul_val)), 1),
        status="ESTABLE" if not es_anomalo else "ANOMALÍA",
        severity_level=severity_level,
        recommended_action=recommended_action,
        failure_code=int(fail_code),
        cause=cause
    )
    
    # Guardar en base de datos
    save_prediction_to_db(motor_id, pred_detail)
    
    return pred_detail

# --- 7. RUTAS DEL API ---

@router.get("/predict/{motor_id}", response_model=PredictionResponse)
def predict_motor(motor_id: int):
    """Predice el estado de salud y RUL para un motor específico."""
    data = get_real_data(motor_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Esperando datos de telemetría para el motor {motor_id}...")
        
    try:
        pred_detail = run_inference_for_motor(motor_id, data)
        last_point = data[-1]
        
        sensor_data = {}
        for k, v in last_point.items():
            try:
                sensor_data[k] = float(v)
            except Exception:
                sensor_data[k] = v
                
        return PredictionResponse(
            sensor_data=sensor_data,
            prediction=pred_detail
        )
    except Exception as e:
        print(f"[ERROR] Error en inferencia IA para motor {motor_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions/history/{motor_id}", response_model=PredictionHistoryResponse)
def get_predictions_history(motor_id: int):
    """Recupera el histórico de predicciones recientes para un motor de InfluxDB."""
    try:
        query = '''
        from(bucket: bucket)
            |> range(start: -24h)
            |> filter(fn: (r) => r["_measurement"] == "predicciones_ia")
            |> filter(fn: (r) => r["id_motor"] == motor_id)
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns:["_time"])
            |> tail(n:50)
        '''
        params = {"bucket": BUCKET, "motor_id": str(motor_id)}
        result = query_api.query(org=ORG, query=query, params=params)
        
        history = []
        for table in result:
            for record in table.records:
                vals = record.values
                t = vals.get("_time")
                t_str = t.strftime("%H:%M:%S") if hasattr(t, "strftime") else str(t)
                history.append(PredictionHistoryPoint(
                    timestamp=t_str,
                    health_score=float(vals.get("health_score") or 0.0),
                    rul_estimated=float(vals.get("rul_estimated") or 0.0),
                    is_anomaly=bool(vals.get("is_anomaly") or False),
                    severity_level=str(vals.get("severity_level") or "ÓPTIMO"),
                    failure_code=int(vals.get("failure_code") or 0),
                    cause=vals.get("cause")
                ))
        return PredictionHistoryResponse(id_motor=motor_id, history=history)
    except Exception as e:
        logger.error(f"Error en predictions_history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fleet/status", response_model=FleetStatusResponse)
def get_fleet_status():
    """Retorna un listado consolidado del estado de todos los motores de la flota activa."""
    try:
        # Obtener los motores activos en la última hora (desde telemetria_maquinaria)
        query_active = '''
        from(bucket: bucket)
            |> range(start: -1h)
            |> filter(fn: (r) => r["_measurement"] == "telemetria_maquinaria")
            |> filter(fn: (r) => r["_field"] == "ciclo")
            |> group(columns: ["id_motor"])
            |> last()
        '''
        params_active = {"bucket": BUCKET}
        result_active = query_api.query(org=ORG, query=query_active, params=params_active)
        active_motor_ids = []
        for table in result_active:
            for record in table.records:
                m_id = record.values.get("id_motor")
                if m_id:
                    active_motor_ids.append(int(m_id))
        
        if not active_motor_ids:
            return FleetStatusResponse(fleet_size=0, severity_counts={}, motors=[])
            
        # Consultar la última predicción para todos los motores en las últimas 24h
        query_all_preds = '''
        from(bucket: bucket)
            |> range(start: -24h)
            |> filter(fn: (r) => r["_measurement"] == "predicciones_ia")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> group(columns: ["id_motor"])
            |> sort(columns:["_time"])
            |> tail(n:1)
        '''
        params_preds = {"bucket": BUCKET}
        result_preds = query_api.query(org=ORG, query=query_all_preds, params=params_preds)
        
        latest_predictions = {}
        for table in result_preds:
            for record in table.records:
                vals = record.values
                m_id = vals.get("id_motor")
                if m_id:
                    t = vals.get("_time")
                    t_str = t.strftime("%Y-%m-%dT%H:%M:%SZ") if hasattr(t, "strftime") else str(t)
                    latest_predictions[int(m_id)] = {
                        "health_score": float(vals.get("health_score") or 0.0),
                        "rul_estimated": float(vals.get("rul_estimated") or 0.0),
                        "severity_level": str(vals.get("severity_level") or "ÓPTIMO"),
                        "timestamp": t_str
                    }
        
        motors_summary = []
        severity_counts = {
            "ÓPTIMO": 0,
            "ATENCIÓN": 0,
            "ALERTA": 0,
            "CRÍTICO": 0,
            "FALLO INMINENTE": 0
        }
        
        for m_id in sorted(active_motor_ids):
            latest_pred = latest_predictions.get(m_id)
            if not latest_pred:
                # Calcular al vuelo si no hay predicción registrada
                data = get_real_data(m_id)
                if data:
                    pred_detail = run_inference_for_motor(m_id, data)
                    last_time = data[-1]["timestamp"]
                    last_time_str = last_time.strftime("%Y-%m-%dT%H:%M:%SZ") if hasattr(last_time, 'strftime') else str(last_time)
                    latest_pred = {
                        "health_score": pred_detail.health_score,
                        "rul_estimated": pred_detail.rul_estimated,
                        "severity_level": pred_detail.severity_level,
                        "timestamp": last_time_str
                    }
            
            if latest_pred:
                sev = latest_pred["severity_level"]
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
                motors_summary.append(MotorFleetInfo(
                    id_motor=m_id,
                    health_score=latest_pred["health_score"],
                    rul_estimated=latest_pred["rul_estimated"],
                    severity_level=sev,
                    last_updated=latest_pred["timestamp"]
                ))
                
        return FleetStatusResponse(
            fleet_size=len(motors_summary),
            severity_counts=severity_counts,
            motors=motors_summary
        )
    except Exception as e:
        logger.error(f"Error en fleet_status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/info", response_model=ModelMetadataResponse)
def get_models_info():
    """Informa sobre el estado de carga y arquitectura de los modelos IA en producción."""
    columns_para_scaler = [
        'ajuste_1', 'ajuste_2', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6', 'sensor_7',
        'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_15',
        'sensor_16', 'sensor_17', 'sensor_20', 'sensor_21',
        'sensor_4_std', 'sensor_4_diff', 'sensor_11_std', 'sensor_11_diff',
        'sensor_15_std', 'sensor_15_diff', 'sensor_21_std', 'sensor_21_diff'
    ]
    return ModelMetadataResponse(
        models_loaded={
            "lstm_regresion_rul": lstm_model is not None,
            "autoencoder_anomalias": autoencoder is not None,
            "xgboost_clasificacion": xgboost_model is not None,
            "scaler": scaler is not None
        },
        input_shape_keras=[1, 30, 24],
        input_shape_xgb=[1, 720],
        scaler_columns=columns_para_scaler
    )

@router.post("/predict/batch")
def predict_batch(req: BatchPredictionRequest):
    """Ejecuta predicciones en lote para una lista de IDs de motores."""
    results = {}
    for motor_id in req.motor_ids:
        data = get_real_data(motor_id)
        if not data:
            results[motor_id] = {"status": "error", "message": "Sin datos"}
            continue
        try:
            pred_detail = run_inference_for_motor(motor_id, data)
            results[motor_id] = pred_detail
        except Exception as e:
            results[motor_id] = {"status": "error", "message": str(e)}
    return {"predictions": results}

# --- 8. ARRANGEMENT ORIGINAL & LEGACY COMPATIBILITY ---

app.include_router(router)

@app.get("/predict")
def legacy_predict():
    """Endpoint legado para compatibilidad con código anterior (por defecto motor 1)."""
    try:
        res = predict_motor(1)
        return res
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
def healthcheck():
    """Endpoint básico para docker compose healthcheck."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)