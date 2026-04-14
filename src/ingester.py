import paho.mqtt.client as mqtt
import json

# Configuración 
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "factory/machine_01/telemetry"

def on_connect(client, userdata, flags, rc, properties=None):
    """Se ejecuta cuando nos conectamos al broker."""
    if rc == 0:
        print(f"✅ Ingester CONECTADO. Escuchando en: {TOPIC}")
        client.subscribe(TOPIC)
    else:
        print(f"❌ Error de conexión: {rc}")

def on_message(client, userdata, msg):
    """Se ejecuta cada vez que llega un dato del sensor."""
    try:
        # 1. Recibir el paquete de datos
        payload = msg.payload.decode()
        data = json.loads(payload)
        
        # 2. Mostrarlo por pantalla (De momento solo imprimimos)
        print(f"📥 DATO RECIBIDO -> Temp: {data['temperature']}°C | Vib: {data['vibration']} | RPM: {data['rpm']}")
        
    except Exception as e:
        print(f"⚠️ Error al procesar mensaje: {e}")

def run_ingester():
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    
    # Asignamos las funciones de respuesta
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(BROKER, PORT)
        # loop_forever() hace que el script se quede "escuchando" sin cerrarse
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n🛑 Ingester detenido por el usuario.")
    finally:
        client.disconnect()

if __name__ == "__main__":
    run_ingester()