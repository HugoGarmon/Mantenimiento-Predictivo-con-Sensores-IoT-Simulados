# 🏭 AI Predictive Maintenance System

Sistema de monitorización industrial en tiempo real con detección de anomalías y predicción de fallos.

## 🚀 Componentes del Módulo de Desarrollo (Semana 1)

### 1. Simulador de Sensores (`src/simulator.py`)
Genera datos sintéticos de telemetría (Vibración, Temperatura, Presión, RPM) y los envía vía MQTT a un broker.

### 2. Ingester de Datos (`src/ingester.py`)
Suscriptor MQTT que recibe la telemetría en tiempo real y procesa los paquetes JSON.

## 🛠️ Cómo ejecutar
1. Crear entorno virtual: `python -m venv venv`
2. Activar: `.\venv\Scripts\activate` (Windows)
3. Instalar dependencias: `pip install paho-mqtt`
4. Ejecutar Ingester: `python src/ingester.py`
5. Ejecutar Simulador: `python src/simulator.py`
