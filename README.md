# 🏭 AI Predictive Maintenance System

Sistema integral de monitorización industrial basado en microservicios para la detección de anomalías y predicción de fallos (RUL) mediante Deep Learning.

## 🏗️ Arquitectura del Sistema

El proyecto utiliza una arquitectura de microservicios contenedorizada con **Docker**:

1.  **Broker MQTT (Mosquitto):** El sistema de mensajería que centraliza el flujo de datos.
2.  **Sensor Simulator (Python):** Generador de telemetría industrial (vibración, temperatura, RPM, presión).
3.  **Data Consumer/Ingester (Python):** Suscriptor MQTT que procesa y persiste los datos en tiempo real.
4.  **Time Series DB (InfluxDB 2):** Base de datos optimizada para el almacenamiento de métricas de sensores.
5.  **Backend API (FastAPI):** Servicio que expone los datos y predicciones de IA.
6.  **Dashboard (Streamlit):** Interfaz visual para la monitorización de activos.

## 🚀 Estado del Desarrollo

### Semana 1: Infraestructura & Ingesta ✅
- [x] Configuración de Docker Compose y Redes.
- [x] Despliegue de Broker MQTT y Base de Datos InfluxDB.
- [x] Desarrollo del Ingester con persistencia real.

### Semana 2: API & Dashboard ✅
- [x] Esqueleto de la API REST con **FastAPI**.
- [x] Integración de la API con los datos reales de InfluxDB.
- [x] Dashboard interactivo con **Streamlit**.

## 🛠️ Guía de Ejecución

### 1. Requisitos previos
- Docker & Docker Desktop.
- Python 3.9+ (para desarrollo local).

### 2. Levantar la infraestructura (Docker)
Desde la raíz del proyecto, ejecuta:
```powershell
# Crear la red si es la primera vez
docker network create shared-network

# Levantar todos los servicios
docker-compose up -d --build