# 🏭 AI Predictive Maintenance System - Industrial IoT

> **Sistema integral de monitorización industrial basado en microservicios** para detección de anomalías y predicción de vida útil (RUL) mediante Deep Learning.

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## 📖 Descripción General

Este proyecto implementa una **solución integral de mantenimiento predictivo** para entornos industriales. Combina IoT en tiempo real, procesamiento de datos y machine learning avanzado para:

✅ **Monitorizar maquinaria** en tiempo real mediante sensores IoT  
✅ **Detectar anomalías** usando redes neuronales (Autoencoder)  
✅ **Predecir fallos** con LSTM estimando vida útil remanente (RUL)  
✅ **Clasificar problemas** mediante XGBoost  
✅ **Visualizar datos** con dashboard interactivo en Streamlit  

---

## ✨ Características Principales

### 🤖 Machine Learning Avanzado
| Modelo | Propósito | Tipo |
|--------|----------|------|
| **Autoencoder** | Cálculo de Health Score (detección de anomalías) | Neural Network |
| **LSTM** | Predicción de Vida Útil Remanente (RUL) | Recurrent Neural Network |
| **XGBoost** | Clasificación de tipos de fallo | Gradient Boosting |

### 📊 Datos en Tiempo Real
- Lectura de **4 sensores industriales**: Temperatura, Vibración, Presión, RPM
- Frecuencia: **Cada 3 segundos**
- Persistencia: **InfluxDB 2** (Time-Series Database optimizada)
- Protocolo: **MQTT** (IoT estándar industrial)

### 🎨 Dashboard Interactivo
- Métricas en vivo: Health Score, RUL, estado de sensores
- Gráficas de tendencias temporales
- Registro de incidencias y anomalías detectadas
- Interfaz reactiva con **Streamlit**

### 🏗️ Arquitectura Escalable
- Microservicios contenedorizados con Docker
- Red Docker Compose para orquestación
- API REST para integraciones externas

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAPA DE PRESENTACIÓN                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Streamlit Dashboard (localhost:8501)                    │   │
│  │  - Health Score, RUL, Tendencias                         │   │
│  │  - Registro de Incidencias en Tiempo Real                │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↑
                    (HTTP REST API)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      CAPA DE LÓGICA                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  FastAPI Backend (localhost:8000)                        │   │
│  │  - Inferencia IA (Autoencoder, LSTM, XGBoost)           │   │
│  │  - Normalización de datos                               │   │
│  │  - Endpoint: GET /predict                               │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↑
                    (HTTP REST Query)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CAPA DE DATOS                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  InfluxDB 2 (localhost:8086)                             │   │
│  │  - Time-Series DB                                        │   │
│  │  - Bucket: 'home'                                        │   │
│  │  - Measurement: 'telemetria_maquinaria'                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↑                                    │
│                   (MQTT Subscribe)                               │
│                              ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Data Consumer (Python)                                  │   │
│  │  - Suscrito a: factory/machine_01/telemetry              │   │
│  │  - Transforma y persiste datos en InfluxDB               │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↑
                    (MQTT Publish)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CAPA DE TELEMETRÍA                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  MQTT Broker - Mosquitto (localhost:1883)                │   │
│  │  - Topic: factory/machine_01/telemetry                   │   │
│  │  - Explorer: localhost:4000                              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↑                                    │
│                   (MQTT Publish)                                 │
│                              ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Sensor Simulator (Python)                               │   │
│  │  - Genera datos sintéticos de máquina industrial          │   │
│  │  - Envía cada 3 segundos                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Flujo de Datos

```
Sensor Simulator → MQTT Broker → Data Consumer → InfluxDB → FastAPI → Streamlit Dashboard
                  (JSON msgs)     (Subscribe)     (Store)   (Query)     (HTTP Requests)
```

---

## 📋 Requisitos Previos

### Software Necesario

```bash
✓ Docker Desktop 4.0+  (Windows/Mac) o Docker Engine (Linux)
✓ Docker Compose 2.0+
✓ Python 3.9+
✓ PowerShell 5.0+ (Windows) o Bash (macOS/Linux)
✓ Git
```

### Hardware Recomendado

```
- RAM: Mínimo 4GB, Recomendado 8GB
- CPU: 2 cores
- Disco: 5GB disponible (para imágenes Docker)
```

### Puertos Requeridos

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| Streamlit | 8501 | Dashboard |
| FastAPI | 8000 | API Backend |
| InfluxDB | 8086 | Time-Series DB |
| Mosquitto MQTT | 1883 | IoT Broker |
| MQTT Web | 9001 | WebSocket MQTT |
| MQTT Explorer | 4000 | UI para MQTT |

---

## 💻 Instalación

### Paso 1: Clonar o Descargar el Proyecto

```bash
cd /c/Proyecto_Carlos
cd industrial-predictive-ai
```

### Paso 2: Verificar la Estructura

```bash
# Windows PowerShell
dir -Recurse -Exclude venv | head -20

# macOS/Linux
ls -la
```

Estructura esperada:
```
industrial-predictive-ai/
├── api/
│   └── main.py              # FastAPI Backend
├── app/
│   └── main.py              # Streamlit Dashboard
├── models/
│   ├── lstm_model.keras     # Modelo LSTM (RUL)
│   ├── autoencoder_model.keras  # Modelo Autoencoder (Health)
│   └── xgboost_model.pkl    # Modelo XGBoost (Fallo)
├── mosquitto/
│   ├── src/
│   │   ├── sensor_simulator.py      # Generador de datos
│   │   ├── data_consumer.py         # Ingester MQTT→InfluxDB
│   │   └── requirements.txt
│   └── mosquitto.conf
├── influxdb/
│   ├── .env.influxdb2-admin-username
│   ├── .env.influxdb2-admin-password
│   └── .env.influxdb2-admin-token
├── compose.yml              # Orquestación Docker
└── README.md
```

### Paso 3: Crear la Red Docker (Primera Vez)

```bash
# Windows PowerShell
docker network create shared-network

# Verificar
docker network ls
```

### Paso 4: Configurar Variables de Entorno

Los archivos de configuración de InfluxDB están en `influxdb/`:

```bash
# Verificar que existen
ls -la influxdb/.env.*

# Contenido típico:
# .env.influxdb2-admin-username → "admin"
# .env.influxdb2-admin-password → "password123"
# .env.influxdb2-admin-token → "token-largo-aqui"
```

⚠️ **IMPORTANTE**: Estos tokens deben coincidir en:
- `compose.yml` (servicios)
- `api/main.py` (línea 26)
- `mosquitto/src/data_consumer.py` (línea 19)

---

## 🚀 Guía de Ejecución

### Opción A: Ejecución Completa (RECOMENDADO)

#### 1. Iniciar Todos los Servicios

```bash
# Cambiar al directorio del proyecto
cd /c/Proyecto_Carlos/industrial-predictive-ai

# Levantar todos los servicios en background
docker-compose up -d --build

# Verificar que todo esté corriendo
docker-compose ps
```

**Salida esperada:**
```
NAME                 STATUS         PORTS
iot-mqtt-broker      Up 2 minutes   1883/tcp, 9001/tcp
mqtt-explorer        Up 2 minutes   0.0.0.0:4000->4000/tcp
iot-sensor-simulator Up 2 minutes   
iot-data-consumer    Up 2 minutes   
influxdb2            Up 2 minutes   0.0.0.0:8086->8086/tcp
```

#### 2. Verificar Logs (Diagnóstico)

```bash
# Ver logs de un servicio específico
docker-compose logs mqtt-broker
docker-compose logs iot-sensor-simulator
docker-compose logs iot-data-consumer
docker-compose logs influxdb2

# Ver logs en tiempo real
docker-compose logs -f iot-data-consumer

# Detener logs
# Presionar Ctrl+C
```

#### 3. Arrancar API FastAPI

En **una nueva terminal/PowerShell**:

```bash
cd /c/Proyecto_Carlos/industrial-predictive-ai

# Activar entorno virtual (si lo tienes)
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install fastapi uvicorn tensorflow joblib influxdb-client

# Ejecutar API
python api/main.py
```

**Salida esperada:**
```
✅ [SISTEMA] Todos los modelos cargados y listos.
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Press CTRL+C to quit
```

✅ **API disponible en**: `http://localhost:8000`

#### 4. Arrancar Dashboard Streamlit

En **otra terminal/PowerShell**:

```bash
cd /c/Proyecto_Carlos/industrial-predictive-ai

# Activar el mismo entorno virtual
# Windows
.\venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate

# Instalar dependencias Streamlit
pip install streamlit requests pandas

# Ejecutar Dashboard
streamlit run app/main.py
```

**Salida esperada:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

✅ **Dashboard disponible en**: `http://localhost:8501`

---

### Opción B: Ejecución sin Backend Contenedorizado (Desarrollo Local)

Si solo quieres correr la API y Dashboard localmente:

```bash
# 1. Levantar solo infraestructura (MQTT + InfluxDB)
docker-compose up -d mqtt-broker influxdb2 mqtt-explorer

# 2. Esperar a que InfluxDB esté listo (10-15 segundos)
sleep 15

# 3. En terminal 1 - Sensor Simulator
docker-compose up iot-sensor-simulator

# 4. En terminal 2 - Data Consumer
docker-compose up iot-data-consumer

# 5. En terminal 3 - API Backend
python api/main.py

# 6. En terminal 4 - Streamlit Dashboard
streamlit run app/main.py
```

---

### Opción C: Detener y Limpiar

```bash
# Detener todos los servicios
docker-compose down

# Ver contenedores detenidos
docker container ls -a

# Limpiar completamente (WARNING: borra datos!)
docker-compose down -v

# Limpiar redes huérfanas
docker network prune -f
```

---

## 📊 Estructura del Proyecto

### `/api` - Backend FastAPI

```python
# api/main.py
- Carga 3 modelos ML (LSTM, Autoencoder, XGBoost)
- GET /predict → Obtiene datos de InfluxDB → Realiza inferencia → Retorna predicciones
- Normalización manual de datos para evitar overflow numérico
- Health Score = 100 - (MSE * 1000)
- RUL Estimado en horas
- Detección de anomalía: health_score < 80%
```

**Respuesta típica del `/predict`:**
```json
{
  "sensor_data": {
    "temperatura": 75.5,
    "vibracion": 3.2,
    "presion": 100.1,
    "rpm": 2500
  },
  "prediction": {
    "health_score": 92.3,
    "is_anomaly": false,
    "rul_estimated": 145.6,
    "status": "ESTABLE",
    "failure_code": 0
  }
}
```

### `/app` - Dashboard Streamlit

```python
# app/main.py
- Conecta a API cada 2 segundos
- Muestra métricas en tiempo real
- Gráficas de tendencias (últimas 30 lecturas)
- Tabla de incidencias (eventos de anomalía)
- Persistencia de datos en session_state de Streamlit
```

### `/models` - Modelos Machine Learning

```
models/
├── lstm_model.keras           (425 KB) → RUL Prediction
├── autoencoder_model.keras    (91 KB)  → Anomaly Detection
└── xgboost_model.pkl         (221 KB) → Fault Classification
```

### `/mosquitto` - IoT Data Pipeline

```
mosquitto/
├── src/
│   ├── sensor_simulator.py      → Genera datos sintéticos cada 3s
│   ├── data_consumer.py         → Consume MQTT → Escribe en InfluxDB
│   └── requirements.txt         → paho-mqtt, influxdb-client
└── mosquitto.conf              → Configuración del broker MQTT
```

### `/influxdb` - Base de Datos Time-Series

```
influxdb/
├── .env.influxdb2-admin-username    → "admin"
├── .env.influxdb2-admin-password    → "paso1234"
└── .env.influxdb2-admin-token       → Token de acceso
```

**Estructura InfluxDB:**
- **Org**: docs
- **Bucket**: home
- **Measurement**: telemetria_maquinaria
- **Tags**: maquina_id
- **Fields**: temperatura, vibracion, presion, rpm

---

## 🔌 API Endpoints

### GET `/predict`

**Descripción**: Obtiene predicción en tiempo real basada en último sensor data

**URL**: `http://localhost:8000/predict`

**Método**: `GET`

**Parámetros**: Ninguno

**Respuesta exitosa (200)**:
```json
{
  "sensor_data": {
    "temperatura": 75.5,
    "vibracion": 3.2,
    "presion": 100.1,
    "rpm": 2500
  },
  "prediction": {
    "health_score": 92.3,
    "is_anomaly": false,
    "rul_estimated": 145.6,
    "status": "ESTABLE",
    "failure_code": 0
  }
}
```

**Respuesta error (esperando datos)**:
```json
{
  "status": "error",
  "message": "Esperando datos de InfluxDB..."
}
```

**Ejemplos de uso**:

```bash
# cURL
curl -X GET http://localhost:8000/predict

# Python
import requests
response = requests.get("http://localhost:8000/predict")
print(response.json())

# PowerShell
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Get
```

### GET `/docs` (Swagger UI)

**URL**: `http://localhost:8000/docs`

Documentación interactiva de la API generada automáticamente por FastAPI.

---

## ⚙️ Configuración

### Cambiar Intervalo de Sensores

**Archivo**: `mosquitto/src/sensor_simulator.py` (línea 46)

```python
time.sleep(3)  # Cambiar a 5, 10, etc.
```

### Cambiar Umbrales de Anomalía

**Archivo**: `api/main.py` (línea 98)

```python
es_anomalo = health_score < 80.0  # Cambiar 80 al valor deseado
```

### Cambiar Tokens InfluxDB

1. **Editar archivos de secrets**:
   ```bash
   influxdb/.env.influxdb2-admin-username
   influxdb/.env.influxdb2-admin-password
   influxdb/.env.influxdb2-admin-token
   ```

2. **Actualizar en `api/main.py`** (línea 26):
   ```python
   TOKEN = "tu-nuevo-token"
   ```

3. **Actualizar en `mosquitto/src/data_consumer.py`** (línea 19):
   ```python
   INFLUX_TOKEN = "tu-nuevo-token"
   ```

4. **Reiniciar servicios**:
   ```bash
   docker-compose down -v
   docker-compose up -d --build
   ```

---

## 🐛 Troubleshooting

### ❌ "Connection refused" en API

**Causa**: API intenta conectar a InfluxDB pero no está corriendo

**Solución**:
```bash
# Verificar que InfluxDB está corriendo
docker-compose ps

# Si no está:
docker-compose up -d influxdb2

# Esperar 10-15 segundos y reintentar
```

### ❌ Streamlit dice "Error de conexión con el Backend"

**Causa**: API no está corriendo o no está en `localhost:8000`

**Solución**:
```bash
# Verificar que API está corriendo
curl http://localhost:8000/docs

# Si falla, iniciar API en terminal separada:
python api/main.py
```

### ❌ "No se reciben datos de sensores"

**Causa**: Sensor Simulator no está conectado al broker

**Solución**:
```bash
# Verificar logs
docker-compose logs iot-sensor-simulator

# Si falla conexión, reiniciar:
docker-compose restart iot-sensor-simulator

# Verificar que MQTT Broker está corriendo
docker-compose ps mqtt-broker
```

### ❌ InfluxDB no guarda datos

**Causa**: Data Consumer no puede escribir o token incorrecto

**Solución**:
```bash
# Ver logs
docker-compose logs iot-data-consumer

# Verificar token en:
influxdb/.env.influxdb2-admin-token

# Reiniciar Consumer con nuevo token:
docker-compose down iot-data-consumer
# Editar token en mosquitto/src/data_consumer.py
docker-compose up -d iot-data-consumer
```

### ❌ "Permission denied" en Windows PowerShell

**Causa**: Política de ejecución de scripts

**Solución**:
```powershell
# Permitir scripts locales
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# O usar cmd.exe en su lugar
cmd.exe /c "python api/main.py"
```

### ❌ Puertos ya en uso

**Causa**: Otro proceso ocupa los puertos (8501, 8000, 8086, 1883)

**Solución Windows**:
```powershell
# Encontrar proceso en puerto
netstat -ano | findstr :8501

# Matar proceso
taskkill /PID <PID> /F
```

**Solución macOS/Linux**:
```bash
# Encontrar proceso
lsof -i :8501

# Matar proceso
kill -9 <PID>
```

### ❌ "Models not found" en API

**Causa**: Falta la carpeta `models/` con archivos `.keras` y `.pkl`

**Solución**:
```bash
# Verificar que existen
ls -la models/

# Si faltan, descargarlos o entrenarlos:
# (Ver documentación de entrenamiento)
```

---


