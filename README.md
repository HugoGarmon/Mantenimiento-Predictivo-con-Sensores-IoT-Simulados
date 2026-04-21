# 🏭 AI Predictive Maintenance System

Sistema integral de monitorización industrial basado en microservicios para la detección de anomalías y predicción de fallos (RUL) mediante Deep Learning.

## 🏗️ Arquitectura del Sistema (Full Stack)

El proyecto utiliza una arquitectura de microservicios contenedorizada con **Docker**:

1.  **Broker MQTT (Mosquitto):** Centro de mensajería para el flujo de telemetría.
2.  **Sensor Simulator (Python):** Generador de datos industriales (vibración, temperatura, RPM, presión).
3.  **Data Consumer/Ingester (Python):** Procesa y persiste datos MQTT en tiempo real.
4.  **Time Series DB (InfluxDB 2):** Almacenamiento y consulta eficiente de métricas.
5.  **Backend API (FastAPI):** Lógica de negocio y servicio de predicciones de IA.
6.  **Dashboard (Streamlit):** Visualización avanzada y estado de salud de activos.

## 🚀 Estado del Desarrollo

### Semana 1: Infraestructura & Ingesta ✅
- [x] Configuración de Docker Compose y Redes.
- [x] Despliegue de Broker MQTT e InfluxDB.
- [x] Desarrollo del Ingester con persistencia real.

### Semana 2: API & Dashboard ✅
- [x] Esqueleto de la API REST con **FastAPI**.
- [x] Integración de la API con los datos reales de InfluxDB.
- [x] Dashboard interactivo inicial con **Streamlit**.

### Semana 3: Inteligencia y Monitorización en Tiempo Real 🚧
- [x] Implementación de lógica de **Health Score** y **RUL** (Mockups de IA).
- [x] Dashboard dinámico con refresco automático y alertas visuales.
- [x] Endpoint `/predict` operativo para consumo de frontend.

## 🛠️ Guía de Ejecución

### 1. Requisitos previos
- Docker & Docker Desktop.
- Python 3.9+ (entorno `venv` activo).

### 2. Levantar la infraestructura (Docker)
Desde la raíz del proyecto:
```powershell
# Crear la red si es la primera vez
docker network create shared-network

# Levantar servicios de datos
docker-compose up -d --build