import paho.mqtt.client as mqtt
import json
import time
import random
from datetime import datetime

# Configuración del Broker Público para pruebas
BROKER = "broker.hivemq.com" 
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
    # Inicializamos el cliente MQTT
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    
    try:
        client.connect(BROKER, PORT)
        print(f"🚀 Simulador ACTIVO. Enviando a: {BROKER}")
        
        while True:
            data = generate_sensor_data()
            payload = json.dumps(data)
            client.publish(TOPIC, payload)
            print(f"📤 Enviado: {payload}")
            time.sleep(3) # Envía cada 3 segundos
            
    except KeyboardInterrupt:
        print("\n🛑 Simulador apagado.")
    finally:
        client.disconnect()

if __name__ == "__main__":
    run_simulator()