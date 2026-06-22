import paho.mqtt.client as mqtt
import json
import os
from datetime import datetime
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import time
from dotenv import load_dotenv

load_dotenv()

# --- Configuración MQTT ---
# Asegúrate de que este nombre sea el que sale en el compose.yml (mosquitto)
BROKER_HOST = os.getenv("BROKER_HOST", "mosquitto") 
BROKER_PORT = int(os.getenv("BROKER_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
TOPIC = "factory/+/telemetry" # Suscribirse a todos los motores usando wildcard

# --- Configuración InfluxDB ---
INFLUX_URL = os.getenv("INFLUXDB_URL", "http://influxdb2:8086")
INFLUX_ORG = os.getenv("INFLUXDB_ORG", "docs")
INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET", "home")
INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN") 


# --- Inicializar Cliente InfluxDB ---
try:
    influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    write_api = influx_client.write_api(write_options=SYNCHRONOUS)
    print("Conexión con InfluxDB 2 preparada.")
except Exception as e:
    print(f"Error inicializando InfluxDB: {e}")

# --- Callbacks MQTT (Actualizados a v2 para evitar el error de argumentos) ---
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"Consumidor conectado a {BROKER_HOST}. Suscribiéndose a: {TOPIC}")
        client.subscribe(TOPIC)
    else:
        print(f"Error de conexión MQTT. Código: {reason_code}")

def on_message(client, userdata, msg):
    try:
        # 1. Decodificar el JSON real que envía tu simulador
        payload = json.loads(msg.payload.decode())
        id_motor = payload.get("id_motor", 1)
        ciclo = payload.get("ciclo", 0)
        
        # 2. Mostrar por consola lo que de verdad llega
        print(f"\n [{datetime.now().strftime('%H:%M:%S')}] {msg.topic}")
        print(f"   📥 DATOS: Motor: {id_motor} | Ciclo: {ciclo} | {len(payload) - 3} variables de sensores y ajustes recibidas")
        
        # 3. Crear el punto de datos para InfluxDB dinámicamente
        punto = Point("telemetria_maquinaria") \
            .tag("maquina_id", f"machine_{id_motor:02d}") \
            .tag("id_motor", str(id_motor))
        
        # Iterar sobre las claves del payload y agregarlas como campos
        for key, val in payload.items():
            if key in ["timestamp", "id_motor"]:
                continue
            
            if key == "ciclo":
                punto = punto.field(key, int(val))
            elif "sensor" in key or "ajuste" in key:
                punto = punto.field(key, float(val))
            else:
                try:
                    punto = punto.field(key, float(val))
                except ValueError:
                    punto = punto.field(key, str(val))
        
        # 4. Escribir en la base de datos
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=punto)
        print("✅ Guardado con éxito en InfluxDB.")
        
    except Exception as e:
        print(f"❌ Error procesando el mensaje: {e}")

# --- Ejecución Principal ---
if __name__ == "__main__":
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "Consumidor_Ingesta")
    client.on_connect = on_connect
    client.on_message = on_message
    
    # Configurar autenticación si se provee
    if MQTT_USER and MQTT_PASSWORD:
        client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
        print(f"🔐 Usando autenticación MQTT para el usuario: {MQTT_USER}")
    
    connected = False
    while not connected:
        try:
            print(f"Conectando al broker en {BROKER_HOST}...")
            client.connect(BROKER_HOST, BROKER_PORT, 60)
            connected = True
        except Exception as e:
            print(f"No se pudo conectar al broker, reintentando en 5s... {e}")
            time.sleep(5)

    client.loop_forever()