from fastapi import FastAPI
from influxdb_client import InfluxDBClient
import uvicorn
import random

app = FastAPI(title="Industrial AI API - Semana 3 Mock")

# Configuración InfluxDB
URL = "http://localhost:8086"
TOKEN = "6wtWMbUQhEJfbEGL-JiVWpF-rL0jidkZnAkvrR1hSaPRTDKeh7zP-ep0NWyeQ3EOzKVgvctIAj8aLas1NQqXYQ=="
ORG = "docs"
BUCKET = "home"

def get_real_data():
    """Consulta el último dato real de InfluxDB"""
    try:
        client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
        query_api = client.query_api()
        query = f'from(bucket: "{BUCKET}") |> range(start: -1m) |> last()'
        result = query_api.query(org=ORG, query=query)
        
        data = {}
        for table in result:
            for record in table.records:
                data[record.get_field()] = record.get_value()
        client.close()
        return data
    except:
        return None

@app.get("/predict")
def predict():
    real_data = get_real_data()
    
    if not real_data:
        return {"status": "error", "message": "No hay datos en InfluxDB"}

    # --- SIMULACIÓN DE IA (Sustituir cuando la compañera termine) ---
    # Simulamos un "Health Score" del 0 al 100
    # Si la temperatura sube de 85 o la vibración de 4.5, el score baja
    temp = real_data.get("temperatura", 0)
    vib = real_data.get("vibracion", 0)
    
    health_score = 100 - (max(0, temp - 70) * 2) - (max(0, vib - 3) * 10)
    health_score = max(0, min(100, health_score)) # Limitar entre 0 y 100

    status = "Saludable"
    if health_score < 40: status = "Fallo Inminente"
    elif health_score < 75: status = "Mantenimiento Necesario"

    return {
        "sensor_data": real_data,
        "prediction": {
            "health_score": round(health_score, 2),
            "status": status,
            "rul_estimated": int(health_score * 0.5) # RUL = Remaining Useful Life (Horas)
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)