import paho.mqtt.client as mqtt
import json
import time
import os
from datetime import datetime, timezone

# --- CONFIGURACIÓN MQTT ---
BROKER = os.getenv("BROKER_HOST", "mosquitto")
PORT = int(os.getenv("BROKER_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
# Se construirá el topic dinámicamente como: factory/machine_{id_motor:02d}/telemetry

DATA_FILE = "test_FD001.txt"

# --- CONFIGURACIÓN SIMULADOR ---
# Velocidad de simulación en milisegundos (por defecto 3000ms = 3s)
SIMULATION_SPEED_MS = int(os.getenv("SIMULATION_SPEED_MS", "3000"))
# Motores activos separados por comas (ej: "1,2,3,4,5"). "*" o vacío simula todos los 100 motores.
ACTIVE_MOTORS_ENV = os.getenv("ACTIVE_MOTORS", "1")

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
    
    # Configurar autenticación si se provee
    if MQTT_USER and MQTT_PASSWORD:
        client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
        print(f"🔐 Usando autenticación MQTT para el usuario: {MQTT_USER}")
        
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
    client.loop_start()
    
    # Cargar datos
    motors_data = load_dataset(DATA_FILE)
    if not motors_data:
        print("⚠ No se pudieron cargar datos del dataset. Usando fallback básico...")
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

    # Resolver motores activos
    all_motor_ids = sorted(list(motors_data.keys()))
    active_motor_ids = []
    
    if ACTIVE_MOTORS_ENV.strip() in ["*", ""]:
        active_motor_ids = all_motor_ids
        print(f"⚙️ Configuración: Simulando TODOS los {len(active_motor_ids)} motores de forma concurrente.")
    else:
        try:
            active_motor_ids = [int(m.strip()) for m in ACTIVE_MOTORS_ENV.split(",") if m.strip()]
            active_motor_ids = [m for m in active_motor_ids if m in motors_data]
            if not active_motor_ids:
                active_motor_ids = [1]
            print(f"⚙️ Configuración: Simulando motores {active_motor_ids} de forma concurrente.")
        except Exception as e:
            print(f"⚠️ Error parseando ACTIVE_MOTORS. Usando motor 1. Detalle: {e}")
            active_motor_ids = [1]

    # Inicializar índices de ciclo para cada motor activo
    # Estructura: {motor_id: {"cycle_idx": 0, "cycles": list_of_cycles}}
    motor_states = {}
    for m_id in active_motor_ids:
        motor_states[m_id] = {
            "cycle_idx": 0,
            "cycles": motors_data[m_id]
        }

    interval_sec = max(0.1, SIMULATION_SPEED_MS / 1000.0)
    print(f"🚀 Simulador ACTIVO. Velocidad: {SIMULATION_SPEED_MS}ms por ciclo. Enviando datos...")
    
    try:
        while True:
            t0 = time.time()
            
            # Publicar un ciclo para cada motor activo en esta iteración
            for m_id in active_motor_ids:
                state = motor_states[m_id]
                cycles = state["cycles"]
                idx = state["cycle_idx"]
                
                # Obtener datos de este ciclo
                row = cycles[idx]
                
                # Crear payload e incluir timestamp
                payload_data = row.copy()
                payload_data["timestamp"] = datetime.now(timezone.utc).isoformat()
                payload_str = json.dumps(payload_data)
                
                # Publicar a topic dinámico
                topic = f"factory/machine_{m_id:02d}/telemetry"
                client.publish(topic, payload_str)
                print(f"📤 Enviado: Motor {m_id:02d} | Ciclo {row['ciclo']} -> {topic}")
                
                # Avanzar ciclo
                idx += 1
                if idx >= len(cycles):
                    print(f"🏁 Motor {m_id:02d} completó todos sus ciclos. Reiniciando simulación...")
                    idx = 0
                state["cycle_idx"] = idx
            
            # Calcular sleep dinámico para mantener el intervalo regular
            elapsed = time.time() - t0
            sleep_time = max(0.01, interval_sec - elapsed)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n🛑 Simulador detenido por el usuario.")
    except Exception as e:
        print(f"❌ Error durante la simulación: {e}")
    finally:
        client.loop_stop()
        client.disconnect()
        print("🔌 Desconectado del broker MQTT.")

if __name__ == "__main__":
    run_simulator()