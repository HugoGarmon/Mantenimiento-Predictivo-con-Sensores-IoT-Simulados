import streamlit as st
import requests
import os

# Configurar API URL
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_BASE = f"{API_URL}/api/v1"

st.markdown("# 🛠️ Configuración e Información del Sistema")
st.markdown("Verifique el estado de las conexiones, consulte metadatos de los modelos de IA en producción e inspeccione la arquitectura.")

# 1. Verificar Conexiones en tiempo real
st.subheader("🌐 Estado de Conexión de la Infraestructura")

col_status1, col_status2 = st.columns(2)

with col_status1:
    # Conexión con FastAPI
    api_connected = False
    try:
        res = requests.get(f"{API_URL}/health", timeout=1.5)
        if res.status_code == 200:
            api_connected = True
            st.success("● API Backend (FastAPI): **CONECTADO**")
        else:
            st.error(f"● API Backend (FastAPI): **ERROR {res.status_code}**")
    except Exception as e:
        st.error(f"● API Backend (FastAPI): **DESCONECTADO** ({e})")

with col_status2:
    # Conexión con InfluxDB vía API Backend
    if api_connected:
        try:
            res_fleet = requests.get(f"{API_BASE}/fleet/status", timeout=1.5)
            if res_fleet.status_code == 200:
                st.success("● Base de Datos (InfluxDB 2): **CONECTADO**")
            else:
                st.error("● Base de Datos (InfluxDB 2): **ERROR DE CONSULTA**")
        except Exception:
            st.error("● Base de Datos (InfluxDB 2): **DESCONECTADO**")
    else:
        st.warning("● Base de Datos (InfluxDB 2): **SIN VERIFICAR** (Requiere Backend activo)")

st.markdown("---")

# 2. Información de Modelos de Inteligencia Artificial
st.subheader("🧠 Metadatos de Modelos de IA en Producción")

if api_connected:
    try:
        res_info = requests.get(f"{API_BASE}/models/info", timeout=2)
        if res_info.status_code == 200:
            info = res_info.json()
            models = info.get("models_loaded", {})
            
            # Tabla de estado de carga
            col_m1, col_m2 = st.columns([1, 2])
            
            with col_m1:
                st.write("**Carga de Modelos:**")
                for name, loaded in models.items():
                    status_icon = "🟢 Cargado" if loaded else "🔴 No encontrado"
                    st.write(f"- `{name}`: {status_icon}")
            
            with col_m2:
                st.write("**Dimensiones y Entrada de Datos:**")
                st.write(f"- Formato entrada Keras (LSTM/Autoencoder): `{info.get('input_shape_keras')}`")
                st.write(f"- Formato entrada XGBoost (Aplanado): `{info.get('input_shape_xgb')}`")
                st.write(f"- Total de características escaladas: `{len(info.get('scaler_columns', []))}` variables")
                
            with st.expander("🔬 Ver listado completo de características procesadas por el Scaler"):
                st.write(info.get("scaler_columns"))
                
        else:
            st.warning("No se pudo obtener información detallada de los modelos.")
    except Exception as e:
        st.error(f"Error cargando metadatos del modelo: {e}")
else:
    st.info("Inicie la API de FastAPI para consultar los metadatos de los modelos.")

st.markdown("---")

# 3. Explicación de la Arquitectura
st.subheader("🏗️ Arquitectura de Mantenimiento Predictivo 4.0")
st.markdown("""
El flujo de datos del sistema está completamente automatizado extremo a extremo:
1. **Simulador de Sensores:** Lee secuencialmente el dataset de la NASA `test_FD001.txt` a la velocidad configurada (`SIMULATION_SPEED_MS`) y publica las lecturas por MQTT.
2. **Mosquitto MQTT Broker:** Canaliza los mensajes en tiempo real utilizando topics dinámicos para cada máquina (`factory/machine_XX/telemetry`).
3. **Data Consumer:** Ingesta los mensajes suscritos vía wildcard (`factory/+/telemetry`), realiza validaciones y escribe las muestras crudas a **InfluxDB 2**.
4. **API Backend (FastAPI):** Proporciona los endpoints que consultan las últimas 30 muestras en InfluxDB, preprocesa los datos (suavizado media móvil, std, diff y normalización) y ejecuta en paralelo los 3 modelos de Inteligencia Artificial:
   * **Autoencoder (LSTM):** Calcula el error de reconstrucción (MSE) de la señal para estimar el **Health Score**.
   * **LSTM Regressor:** Predice la **Vida Útil Restante (RUL)** esperada en ciclos.
   * **XGBoost Classifier:** Evalúa el riesgo de fallo en los próximos 30 ciclos.
   * **Base de Datos Histórica:** La API almacena los resultados en `predicciones_ia` en InfluxDB.
5. **Dashboard Multi-Página (Streamlit):** Visualiza los KPIs de la flota y diagnósticos del motor de forma interactiva con actualizaciones automáticas.
""")
