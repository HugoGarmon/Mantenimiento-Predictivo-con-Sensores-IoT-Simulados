import paho.mqtt.client as mqtt
import json
import time
import os
from datetime import datetime

# --- CONFIGURACIÓN MQTT ---
BROKER = os.getenv("BROKER_HOST", "mosquitto")
PORT = int(os.getenv("BROKER_PORT", "1883"))
TOPIC = "factory/machine_01/telemetry"

DATA_FILE = "test_FD001.txt"

def load_dataset(file_path):
    """Carga y procesa el archivo de datos C-MAPSS."""
    if not os.path.exists(file_path):
        print(f"❌ Archivo no encontrado en: {os.path.abspath(file_path)}")
        return {}
    
    print(f"📖 Cargando datos desde {file_path}...")
    motors_data = {}
    try:
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 26:
                    continue
                
                id_motor = int(parts[0])
                ciclo = int(parts[1])
                
                # Crear diccionario para esta fila
                row = {
                    "id_motor": id_motor,
                    "ciclo": ciclo,
                    "ajuste_1": float(parts[2]),
                    "ajuste_2": float(parts[3]),
                    "ajuste_3": float(parts[4])
                }
                
                # Agregar sensores 1 al 21
                for idx in range(1, 22):
                    row[f"sensor_{idx}"] = float(parts[4 + idx])
                
                if id_motor not in motors_data:
                    motors_data[id_motor] = []
                motors_data[id_motor].append(row)
                
        print(f"✅ Se cargaron {len(motors_data)} motores en total.")
        return motors_data
    except Exception as e:
        print(f"❌ Error al leer el dataset: {e}")
        return {}

def run_simulator():
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    
    # Intentar conectar al broker MQTT
    connected = False
    while not connected:
        try:
            print(f"🔄 Intentando conectar al broker MQTT en: {BROKER}:{PORT}...")
            client.connect(BROKER, PORT)
            connected = True
        except Exception as e:
            print(f"⏳ Broker no disponible, reintentando en 5s... ({e})")
            time.sleep(5)

    print("🔌 Conectado al broker MQTT exitosamente.")
    
    # Cargar datos
    motors_data = load_dataset(DATA_FILE)
    if not motors_data:
        print("⚠ No se pudieron cargar datos del dataset. Usando fallback básico...")
        # Fallback simple por si no existe el archivo
        motors_data = {
            1: [
                {
                    "id_motor": 1,
                    "ciclo": c,
                    "ajuste_1": 0.0,
                    "ajuste_2": 0.0,
                    "ajuste_3": 100.0,
                    **{f"sensor_{i}": 500.0 + c * 0.1 for i in range(1, 22)}
                }
                for c in range(1, 101)
            ]
        }

    motor_ids = sorted(list(motors_data.keys()))
    motor_idx = 0
    cycle_idx = 0
    
    print(f"🚀 Simulador ACTIVO. Enviando datos a {TOPIC}...")
    
    try:
        while True:
            current_motor_id = motor_ids[motor_idx]
            motor_cycles = motors_data[current_motor_id]
            
            # Obtener el registro del ciclo actual
            row = motor_cycles[cycle_idx]
            
            # Crear payload incluyendo el timestamp
            payload_data = row.copy()
            payload_data["timestamp"] = datetime.now().isoformat()
            
            payload_str = json.dumps(payload_data)
            client.publish(TOPIC, payload_str)
            print(f"📤 Enviado: Motor {current_motor_id} | Ciclo {row['ciclo']} | Sensores transmitidos")
            
            # Avanzar al siguiente ciclo/motor
            cycle_idx += 1
            if cycle_idx >= len(motor_cycles):
                print(f"🏁 Motor {current_motor_id} completó todos sus ciclos de simulación.")
                cycle_idx = 0
                # Pasar al siguiente motor
                motor_idx = (motor_idx + 1) % len(motor_ids)
                print(f"🔄 Cambiando a Motor {motor_ids[motor_idx]}...")
                
            time.sleep(3) # Envía cada 3 segundos
            
    except KeyboardInterrupt:
        print("\n🛑 Simulador detenido por el usuario.")
    except Exception as e:
        print(f"❌ Error durante la simulación: {e}")
    finally:
        client.disconnect()

if __name__ == "__main__":
    run_simulator()