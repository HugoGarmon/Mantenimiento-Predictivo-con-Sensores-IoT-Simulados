import paho.mqtt.client as mqtt
import json
import os
from datetime import datetime
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import time

# --- Configuración MQTT ---
# Asegúrate de que este nombre sea el que sale en el compose.yml (mosquitto)
BROKER_HOST = os.getenv("BROKER_HOST", "mosquitto") 
BROKER_PORT = int(os.getenv("BROKER_PORT", "1883"))
TOPIC = "factory/machine_01/telemetry" # Asegúrate de que coincida con el simulador

# --- Configuración InfluxDB ---
INFLUX_URL = "http://influxdb2:8086"
INFLUX_ORG = "docs"
INFLUX_BUCKET = "home"
INFLUX_TOKEN = "6wtWMbUQhEJfbEGL-JiVWpF-rL0jidkZnAkvrR1hSaPRTDKeh7zP-ep0NWyeQ3EOzKVgvctIAj8aLas1NQqXYQ==" 


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
        
        # 2. Mostrar por consola lo que de verdad llega
        print(f"\n [{datetime.now().strftime('%H:%M:%S')}] {msg.topic}")
        print(f"   📥 DATOS: Temp: {payload.get('temperature')} | Vib: {payload.get('vibration')}")
        
        # 3. Crear el punto de datos para InfluxDB con los nombres correctos
        punto = Point("telemetria_maquinaria") \
            .tag("maquina_id", "machine_01") \
            .field("temperatura", float(payload.get("temperature"))) \
            .field("vibracion", float(payload.get("vibration"))) \
            .field("presion", float(payload.get("pressure"))) \
            .field("rpm", int(payload.get("rpm")))
        
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