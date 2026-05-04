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
    """Simula datos de una máquina industrial con anomalías más agresivas.
    - 25% de probabilidad de anomalía.
    - Dentro de anomalías, ~12% son catastróficas (picos muy altos/bajos).
    """
    # Probabilidades
    anomaly_chance = random.random() < 0.15

    if anomaly_chance:
        # 12% de las anomalías serán catastróficas
        catastrophic = random.random() < 0.12
        anomaly_type = random.choice(['temperature', 'vibration', 'pressure', 'rpm'])

        if catastrophic:
            print(f"⚠️  🚨 ANOMALÍA CATASTRÓFICA: {anomaly_type.upper()} 🚨")
            if anomaly_type == 'temperature':
                temperature = round(random.uniform(200.0, 500.0), 2)  # pico extremo
                vibration = round(random.uniform(2.5, 4.8), 2)
                pressure = round(random.uniform(95.0, 105.0), 2)
                rpm = random.randint(1800, 3200)
            elif anomaly_type == 'vibration':
                vibration = round(random.uniform(20.0, 60.0), 2)    # vibración muy alta
                temperature = round(random.uniform(65.0, 90.0), 2)
                pressure = round(random.uniform(95.0, 105.0), 2)
                rpm = random.randint(1800, 3200)
            elif anomaly_type == 'pressure':
                # puede ser caída drástica o pico enorme
                if random.random() < 0.5:
                    pressure = round(random.uniform(0.0, 30.0), 2)   # caída
                else:
                    pressure = round(random.uniform(300.0, 600.0), 2) # pico
                temperature = round(random.uniform(65.0, 90.0), 2)
                vibration = round(random.uniform(2.5, 4.8), 2)
                rpm = random.randint(1800, 3200)
            else:  # rpm
                rpm = random.randint(8000, 20000)  # sobre-rev
                temperature = round(random.uniform(65.0, 90.0), 2)
                vibration = round(random.uniform(2.5, 4.8), 2)
                pressure = round(random.uniform(95.0, 105.0), 2)
        else:
            print(f"⚠️  🚨 ANOMALÍA: {anomaly_type.upper()} EN RANGO EXTREMO 🚨")
            # Anomalía moderada
            vibration = round(random.uniform(8.5, 20.0), 2) if anomaly_type == 'vibration' else round(random.uniform(2.5, 4.8), 2)
            temperature = round(random.uniform(110.0, 180.0), 2) if anomaly_type == 'temperature' else round(random.uniform(65.0, 90.0), 2)
            pressure = round(random.uniform(140.0, 220.0), 2) if anomaly_type == 'pressure' else round(random.uniform(95.0, 105.0), 2)
            rpm = random.randint(4500, 7000) if anomaly_type == 'rpm' else random.randint(1800, 3200)
    else:
        # Valores normales
        vibration = round(random.uniform(2.5, 4.8), 2)
        temperature = round(random.uniform(65.0, 90.0), 2)
        pressure = round(random.uniform(95.0, 105.0), 2)
        rpm = random.randint(1800, 3200)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "vibration": vibration,
        "temperature": temperature,
        "pressure": pressure,
        "rpm": rpm
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