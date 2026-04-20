from fastapi import FastAPI
from influxdb_client import InfluxDBClient
import uvicorn

app = FastAPI(title="Industrial AI API - Real Data")

# Configuración de conexión (Igual que el Ingester)
URL = "http://localhost:8086" # Desde fuera de Docker usamos localhost
TOKEN = "my-super-secret-auth-token"
ORG = "docs"
BUCKET = "home"

@app.get("/last-telemetry")
def get_last_telemetry():
    """Consulta el último dato real guardado en InfluxDB."""
    client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
    query_api = client.query_api()

    # Consulta Flux para traer el último registro
    query = f'from(bucket: "{BUCKET}") |> range(start: -1m) |> last()'
    
    result = query_api.query(org=ORG, query=query)
    
    output = {}
    for table in result:
        for record in table.records:
            output[record.get_field()] = record.get_value()
    
    client.close()
    return output

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)