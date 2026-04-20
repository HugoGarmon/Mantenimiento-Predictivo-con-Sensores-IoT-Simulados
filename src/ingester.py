import paho.mqtt.client as mqtt
import json
import os
import time
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# --- CONFIGURACIÓN DE LA COMPAÑERA (DOCKER) ---
# Usamos los nombres de los servicios definidos en el compose.yml
INFLUX_URL = "http://influxdb2:8086"
INFLUX_TOKEN = "my-super-secret-auth-token"  # El que ella puso en secrets
INFLUX_ORG = "docs"
INFLUX_BUCKET = "home"

# Configuración MQTT: Ahora el host es 'mosquitto' (el nombre del servicio en Docker)
BROKER = os.getenv("BROKER_HOST", "mosquitto")
PORT = 1883
TOPIC = "factory/machine_01/telemetry"

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"✅ Ingester CONECTADO al broker '{BROKER}'. Escuchando en: {TOPIC}")
        client.subscribe(TOPIC)
    else:
        print(f"❌ Error de conexión al broker MQTT: {rc}")

def on_message(client, userdata, msg):
    """Recibe datos MQTT y los persiste en InfluxDB."""
    try:
        # 1. Decodificar JSON
        payload = msg.payload.decode()
        data = json.loads(payload)
        
        # 2. Conectar y escribir en InfluxDB
        with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as db_client:
            write_api = db_client.write_api(write_options=SYNCHRONOUS)
            
            # Creamos el "Point" (el registro para la DB)
            point = Point("mediciones_sensores") \
                .tag("maquina", "maquina_01") \
                .field("temperatura", float(data['temperature'])) \
                .field("vibracion", float(data['vibration'])) \
                .field("presion", float(data['pressure'])) \
                .field("rpm", int(data['rpm']))
            
            write_api.write(bucket=INFLUX_BUCKET, record=point)
            
        print(f"📥 Guardado en InfluxDB -> Temp: {data['temperature']}°C | Vib: {data['vibration']}")
        
    except Exception as e:
        print(f"⚠️ Error al procesar o guardar mensaje: {e}")

def run_ingester():
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    # Pequeño delay para asegurar que el broker y la DB hayan arrancado en Docker
    time.sleep(5)

    try:
        client.connect(BROKER, PORT)
        client.loop_forever()
    except Exception as e:
        print(f"❌ Error fatal en el Ingester: {e}")
    finally:
        client.disconnect()

if __name__ == "__main__":
    run_ingester()