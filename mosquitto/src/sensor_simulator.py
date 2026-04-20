import paho.mqtt.client as mqtt
import json
import time
import random
import os
from datetime import datetime

# --- CONFIGURACIÓN PARA DOCKER ---
# Usamos el nombre del servicio en el compose.yml. Si falla, intenta con 'localhost'
BROKER = os.getenv("BROKER_HOST", "mosquitto") 
PORT = 1883
TOPIC = "factory/machine_01/telemetry"

def generate_sensor_data():
    """Simula datos de una máquina industrial."""
    return {
        "timestamp": datetime.now().isoformat(),
        "vibration": round(random.uniform(2.5, 4.8), 2),
        "temperature": round(random.uniform(65.0, 90.0), 2),
        "pressure": round(random.uniform(95.0, 105.0), 2),
        "rpm": random.randint(1800, 3200)
    }

def run_simulator():
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    
    # Bucle de conexión: reintenta hasta que el broker esté arriba
    connected = False
    while not connected:
        try:
            print(f"🔄 Intentando conectar al broker MQTT en: {BROKER}...")
            client.connect(BROKER, PORT)
            connected = True
        except Exception as e:
            print(f"⏳ Broker no disponible, reintentando en 5s... ({e})")
            time.sleep(5)

    print(f"🚀 Simulador ACTIVO. Enviando datos a {TOPIC}...")
    
    try:
        while True:
            data = generate_sensor_data()
            payload = json.dumps(data)
            client.publish(TOPIC, payload)
            print(f"📤 Enviado: {payload}")
            time.sleep(3) # Envía cada 3 segundos
            
    except KeyboardInterrupt:
        print("\n🛑 Simulador apagado.")
    except Exception as e:
        print(f"❌ Error durante la simulación: {e}")
    finally:
        client.disconnect()

if __name__ == "__main__":
    run_simulator()