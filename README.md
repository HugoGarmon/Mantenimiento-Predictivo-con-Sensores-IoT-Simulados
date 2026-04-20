# 🏭 AI Predictive Maintenance System

Sistema integral de monitorización industrial con detección de anomalías y predicción de fallos (RUL) mediante Deep Learning.

## 🏗️ Arquitectura del Proyecto

El sistema se divide en cuatro módulos principales:
1.  **Simulación e Ingesta:** Generación de datos IoT vía MQTT.
2.  **Backend (API):** Servidor FastAPI que centraliza el acceso a los modelos de IA.
3.  **Frontend (Dashboard):** Panel interactivo en Streamlit para visualización en tiempo real.
4.  **Modelos IA:** Autoencoders y LSTMs para análisis predictivo (en desarrollo).

## 🚀 Estado del Desarrollo

### Semana 1: Infraestructura Base ✅
- [x] Estructura de carpetas y repositorio Git.
- [x] Simulador de sensores MQTT (`src/simulator.py`).
- [x] Script de ingesta de telemetría (`src/ingester.py`).

### Semana 2: Backend y Dashboard 🚧
- [x] Esqueleto de la API REST con **FastAPI** (`api/main.py`).
- [x] Interfaz de usuario con **Streamlit** (`app/main.py`).
- [ ] Conexión real Ingester -> Base de Datos -> API.

## 🛠️ Guía de Ejecución

### 1. Preparación del Entorno
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt  # (O instalar: paho-mqtt fastapi uvicorn streamlit requests)