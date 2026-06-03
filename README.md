# 🏭 Sistema End-to-End de Mantenimiento Predictivo con IoT y Deep Learning

> **Proyecto de Nivel de Producción:** Plataforma en tiempo real que implementa un pipeline completo de ingesta IoT (MQTT, InfluxDB) y una API de inferencia basada en Deep Learning (Autoencoder, LSTM, XGBoost) para predecir fallos y estimar la vida útil restante (RUL) de turbinas industriales, usando el dataset de la NASA.

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-1572B6?logo=scikitlearn&logoColor=white)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![InfluxDB](https://img.shields.io/badge/InfluxDB-2.0-22ADF6?logo=influxdb&logoColor=white)](https://www.influxdata.com/)

---

## 📖 Descripción General

Este proyecto implementa una arquitectura en microservicios diseñada para simular la telemetría de turbinas de aviación (NASA C-MAPSS FD001) y procesar dichos datos en tiempo real. Combina tecnologías de mensajería industrial, almacenamiento en series temporales y múltiples modelos de Inteligencia Artificial para ofrecer diagnósticos y pronósticos de salud de maquinaria de forma reactiva y escalable.

### Capacidades del Sistema:
*   **Simulación Industrial:** Stream en tiempo real de los datos reales del dataset de la NASA (21 sensores y 3 ajustes operativos), transmitidos ciclo a ciclo.
*   **Ingesta y Persistencia IoT:** Flujo robusto mediante un Broker MQTT (Eclipse Mosquitto) y un consumidor de datos asíncrono hacia una base de datos temporal InfluxDB.
*   **Inferencia Predictiva (IA):** API REST optimizada que preprocesa secuencias de telemetría y realiza predicciones simultáneas con 3 modelos (Autoencoder LSTM, LSTM Regressor y clasificador XGBoost).
*   **Visualización Ejecutiva:** Panel interactivo en Streamlit con visualizaciones en vivo de métricas críticas, registro de anomalías y tendencias históricas.

---

## 🏗️ Arquitectura del Sistema y Flujo de Datos

El flujo de información recorre todo el espectro de una infraestructura IoT industrial moderna:

```
┌──────────────────────────────┐          ┌──────────────────────────────┐
│  Simulador Sensor (Python)   │ ──MQTT──>│    Broker MQTT (Mosquitto)   │
│  (Lectura test_FD001.txt)    │          │  ( factory/machine_01/telem )│
└──────────────────────────────┘          └──────────────────────────────┘
                                                          │
                                                      (Subscribe)
                                                          ▼
┌──────────────────────────────┐          ┌──────────────────────────────┐
│      InfluxDB 2 (TSDB)       │ <─────── │    Data Consumer (Python)    │
│  (Guardado dinámico campos)  │          │    (Casteo e ingesta segura) │
└──────────────────────────────┘          └──────────────────────────────┘
               │
          (REST Query)
               ▼
┌──────────────────────────────┐          ┌──────────────────────────────┐
│   FastAPI Inferencia Server  │ ───────> │  Streamlit Web Dashboard     │
│   (Escalador + Modelos IA)   │  (HTTP)  │  (Métricas y Gráficas Live)  │
└──────────────────────────────┘          └──────────────────────────────┘
```

---

## 📊 Especificación de Variables (Dataset NASA C-MAPSS)

El pipeline de datos y los modelos de IA se alimentan del conjunto de datos **C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** de la NASA. El sistema utiliza **24 variables activas** en su matriz tridimensional final:

### ⚙️ Ajustes Operativos (Settings)
*   `ajuste_1`: Altitud (Activo)
*   `ajuste_2`: Número de Mach (Activo)
*   `ajuste_3`: TRA (Throttle Resolver Angle) (*Constante - Eliminada*)

### 🌡️ Sensores C-MAPSS FD001
| Variable | Descripción Físico-Técnica | Estado |
|----------|----------------------------|--------|
| `sensor_1` | Temperatura en la entrada del Fan (K) | *Constante - Eliminada* |
| `sensor_2` | Temperatura a la salida del compresor de baja presión (LPC) (K) | **Activo** |
| `sensor_3` | Temperatura a la salida del compresor de alta presión (HPC) (K) | **Activo** |
| `sensor_4` | Temperatura a la salida de la turbina de baja presión (LPT) (K) | **Activo** |
| `sensor_5` | Presión en la entrada del Fan (psia) | **Activo** |
| `sensor_6` | Presión de bypass en el ducto (psia) | **Activo** |
| `sensor_7` | Presión total en la salida del HPC (psia) | **Activo** |
| `sensor_8` | Velocidad física del Fan (rpm) | **Activo** |
| `sensor_9` | Velocidad física del núcleo (rpm) | **Activo** |
| `sensor_10` | Relación de presión del motor (P15/P2) | *Constante - Eliminada* |
| `sensor_11` | Presión estática a la salida del HPC (psia) | **Activo** |
| `sensor_12` | Relación de flujo de combustible a Ps30 (pps/psi) | **Activo** |
| `sensor_13` | Velocidad corregida del Fan (rpm) | **Activo** |
| `sensor_14` | Velocidad corregida del núcleo (rpm) | *Redundante (Corr > 0.98) - Eliminada* |
| `sensor_15` | Relación de bypass | **Activo** |
| `sensor_16` | Eficiencia de la cámara de combustión | *Constante - Eliminada* |
| `sensor_17` | Entalpía de purga | **Activo** |
| `sensor_18` | Velocidad nominal demandada del Fan (rpm) | *Constante - Eliminada* |
| `sensor_19` | Velocidad corregida demandada del Fan (rpm) | *Constante - Eliminada* |
| `sensor_20` | Purga de refrigeración del HPT (lbm/s) | **Activo** |
| `sensor_21` | Purga de refrigeración del LPT (lbm/s) | **Activo** |

### 🛠️ Características de Ingeniería (Feature Engineering)
Para capturar vibración, fatiga y velocidad de degradación, calculamos en una ventana móvil de 10 ciclos las siguientes **8 variables sintéticas**:
- **Desviación Estándar Móvil (STD):** `sensor_4_std`, `sensor_11_std`, `sensor_15_std`, `sensor_21_std`.
- **Diferencia Temporal (Diff a 5 ciclos):** `sensor_4_diff`, `sensor_11_diff`, `sensor_15_diff`, `sensor_21_diff`.

---

## 🤖 Pipeline de Ciencia de Datos e Modelos de IA

### 🔄 Pipeline de Preparación e Inferencia
1.  **Detección y Limpieza:** Se eliminan los 5 sensores constantes y el ajuste operativo 3. Se remueve la multicolinealidad eliminando el `sensor_14` (correlación lineal > 0.98 con `sensor_9`).
2.  **Suavizado de Señal:** Se aplica una media móvil con ventana de 10 ciclos para mitigar el ruido blanco aleatorio.
3.  **Normalización Min-Max:** Se escala todo al rango `[0, 1]` utilizando el `scaler.pkl` ajustado en el entrenamiento.
4.  **Matriz Temporal:** Se empaqueta en tensores de forma `(1, 30, 24)` (ventanas temporales de 30 ciclos de historia por 24 sensores/features).

---

### 🧠 Modelos de Machine Learning y Métricas de Evaluación

El sistema emplea tres modelos especializados entrenados de forma integrada:

#### 1. Autoencoder LSTM (Detección de Anomalías / Health Score)
*   **Arquitectura:** Red neuronal profunda no supervisada. Comprime la ventana temporal a un espacio latente (`LSTM 16` -> `Dense 8`) y reconstruye el tensor de entrada original.
*   **Funcionamiento:** Evaluado en motores sanos. A medida que la turbina se degrada, el error medio absoluto (MAE) de reconstrucción aumenta.
*   **Health Score:** Calculado en producción como `100 - (MAE * 1000)`, con umbral de anomalía establecido en `< 80%`.

#### 2. LSTM Regressor (Predicción de RUL)
*   **Arquitectura:** Red Neuronal Recurrente profunda (`LSTM 64` -> `Dropout 0.3` -> `LSTM 32` -> `Dense 16` -> `Linear`).
*   **Objetivo:** Predecir la Vida Útil Restante (RUL - Remaining Useful Life) medida en ciclos (u horas simuladas).
*   **Resultados de la Evaluación:**
    -   **MAE (Error Medio Absoluto):** **18.94 ciclos** (el modelo estima el momento de fallo con menos de 19 ciclos de error promedio).
    -   **R² (Coeficiente de Determinación):** **0.7995** (el modelo explica el 80% de la varianza en la degradación de la máquina).
    -   *Comparado y seleccionado por encima de arquitecturas GRU (MAE: 18.94, R²: 0.7890) y CNN-1D (MAE: 20.72, R²: 0.7501).*

#### 3. XGBoost Classifier (Clasificación de Estado Crítico / RUL <= 30)
*   **Arquitectura:** Gradient Boosted Trees aplanando la ventana temporal a un vector de 720 características (`30 * 24`).
*   **Objetivo:** Clasificar si la máquina entrará en fallo inminente en los siguientes 30 ciclos.
*   **Métricas de Clasificación (Conjunto de Test):**
    -   **Clase Crítica (RUL <= 30):** Precisión: **93%** | Recall: **98%** | F1-Score: **95%**
    -   **Clase Sana (RUL > 30):** Precisión: **99%** | Recall: **98%** | F1-Score: **99%**
    -   **Exactitud Global (Accuracy):** **98%**

---

## 💻 Requisitos e Instalación Rápida

### 📋 Requisitos de Sistema
*   Docker Desktop & Docker Compose.
*   Python 3.9 o superior.
*   Puertos libres en localhost: `8000` (FastAPI), `8501` (Streamlit), `8086` (InfluxDB) y `1883` (Mosquitto).

### 🚀 Despliegue en 3 Pasos

#### 1. Iniciar Infraestructura Docker
Levanta los contenedores en segundo plano (MQTT, InfluxDB, Sensor Simulator, Data Consumer):
```bash
docker network create shared-network
docker compose up -d --build
```

#### 2. Levantar API FastAPI
Activa el entorno virtual e inicia el backend de inferencia local:
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING='utf-8'
pip install -r requirements.txt
python api/main.py

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python api/main.py
```
*(Espera a que imprima `[SISTEMA] Todos los modelos y el scaler fueron cargados exitosamente.`)*

#### 3. Levantar Dashboard Streamlit
En otra terminal activa, arranca la capa web interactiva:
```bash
# Activar entorno virtual y ejecutar:
streamlit run app/main.py
```
Accede al panel en tu navegador en: [http://localhost:8501](http://localhost:8501).

---

## 💼 Perfil del Proyecto para CV y LinkedIn (Showcase)

Este desarrollo destaca como un proyecto estrella porque abarca disciplinas de **Data Engineering, Data Science y MLOps**. Aquí tienes plantillas listas para incluirlo en tu perfil profesional:

### 📄 Cómo incluirlo en tu Currículum Vitae (CV)

**Título:** Ingeniero de Machine Learning / Arquitecto de Datos IoT
**Proyecto:** Plataforma End-to-End de Mantenimiento Predictivo con Deep Learning (IoT)
*   Diseñé e implementé una arquitectura de microservicios contenedorizados con **Docker Compose**, integrando un flujo de telemetría IoT en tiempo real mediante un broker **MQTT (Mosquitto)** e ingesta directa a la base de datos temporal **InfluxDB**.
*   Entrené y optimicé múltiples arquitecturas de Deep Learning en **TensorFlow/Keras** para monitorización de activos industriales: un **Autoencoder LSTM** para calcular el Health Score de turbinas de aviación (C-MAPSS de la NASA) y un **LSTM Regresor** que estima la vida útil restante (RUL) con un MAE de **18.9 ciclos** ($R^2=0.80$).
*   Desarrollé un clasificador basado en **XGBoost** que predice fallos inminentes a 30 ciclos vista con un **Recall del 98%** y un **F1-Score del 95%** en estados críticos.
*   Construí una **API REST (FastAPI)** para procesamiento y normalización temporal de ventanas tridimensionales de sensores en producción, consumida por un cuadro de mando ejecutivo desarrollado en **Streamlit**.
*   **Stack Tecnológico:** Python, TensorFlow, XGBoost, FastAPI, Streamlit, InfluxDB, MQTT, Docker, MLflow.

---

### 🔗 Publicación sugerida para LinkedIn

```text
🚀 ¡Comparto mi último proyecto estrella de Mantenimiento Predictivo Industrial End-to-End!

He desarrollado una plataforma de microservicios diseñada para monitorizar y predecir fallos en turbinas industriales en tiempo real utilizando Inteligencia Artificial y datos de telemetría de sensores.

Destacables técnicos del proyecto:
1️⃣ Capa de Ingesta IoT: Simulación y publicación de telemetría real (C-MAPSS NASA) mediante un Broker MQTT (Mosquitto) persistiendo en InfluxDB 2 de forma asíncrona.
2️⃣ Procesamiento Temporal: Un pipeline que transforma flujos continuos en secuencias tridimensionales de 30 ciclos para alimentar modelos IA en producción.
3️⃣ Modelos de Machine & Deep Learning:
   - Autoencoder LSTM: Calcula en vivo el "Health Score" de las turbinas basado en errores de reconstrucción.
   - LSTM Regresor: Estima la vida útil restante (RUL) de la máquina con un MAE de solo ~18 ciclos.
   - XGBoost Classifier: Clasifica estados críticos (fallos en menos de 30 ciclos) con un Recall del 98%.
4️⃣ Visualización: Frontend interactivo en Streamlit conectado por API REST a un backend FastAPI que orquesta la inferencia paralela.

Este desarrollo demuestra cómo las técnicas avanzadas de Deep Learning se integran con arquitecturas robustas de ingeniería de datos (MLOps) para generar valor real de negocio y evitar paradas no planificadas en fábricas.

👉 Todo el código fuente y especificaciones técnicas están disponibles en mi repositorio de GitHub: https://github.com/HugoGarmon/Mantenimiento-Predictivo-con-Sensores-IoT-Simulados

#MachineLearning #DeepLearning #IoT #MLOps #DataEngineering #Python #TensorFlow #FastAPI #Streamlit
```
