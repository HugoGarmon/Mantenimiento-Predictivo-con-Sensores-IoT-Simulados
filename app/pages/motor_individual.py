import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta

import os

# Configurar API URL
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_BASE = f"{API_URL}/api/v1"

st.markdown("# ⚙️ Diagnóstico Detallado del Motor")
st.markdown("Monitorización en tiempo real de sensores C-MAPSS y diagnóstico predictivo por Inteligencia Artificial.")

# 1. Obtener la lista de motores activos para el selector
active_motors = [1]
try:
    res_fleet = requests.get(f"{API_BASE}/fleet/status", timeout=2)
    if res_fleet.status_code == 200:
        active_motors = [m["id_motor"] for m in res_fleet.json().get("motors", [])]
        if not active_motors:
            active_motors = [1]
except Exception:
    active_motors = [1]

# Asegurar que el motor previamente seleccionado esté en la lista
selected_default = st.session_state.get("selected_motor_id", 1)
if selected_default not in active_motors:
    active_motors.append(selected_default)
active_motors = sorted(list(set(active_motors)))

# Selector del motor en la barra lateral o al inicio
motor_id = st.selectbox("Seleccione el motor a monitorizar:", active_motors, index=active_motors.index(selected_default))
st.session_state.selected_motor_id = motor_id

# Iniciar fragmento para actualizar en tiempo real
@st.fragment(run_every=timedelta(seconds=2))
def render_motor_details(m_id):
    try:
        # A. Obtener predicción actual
        res = requests.get(f"{API_BASE}/predict/{m_id}", timeout=2)
        if res.status_code != 200:
            st.error(f"Error al conectar con la API para el motor {m_id:02d}.")
            return
            
        payload = res.json()
        data = payload["sensor_data"]
        pred = payload["prediction"]
        
        # B. Título de Ciclo y Motor
        st.subheader(f"Motor Simulado: {m_id:02d} | Ciclo Activo: {int(data.get('ciclo', 0))}")
        
        # C. Alertas Dinámicas y Acciones Recomendadas
        sev = pred["severity_level"]
        rec_action = pred["recommended_action"]
        
        if sev == "ÓPTIMO":
            st.success(f"✅ **SISTEMA OPERANDO EN PARÁMETROS ÓPTIMOS** — {rec_action}")
        elif sev == "ATENCIÓN":
            st.warning(f"🟡 **SEVERIDAD: ATENCIÓN** — {rec_action}")
        elif sev == "ALERTA":
            st.warning(f"🟠 **SEVERIDAD: ALERTA (Degradación Activa)** — {rec_action}")
        elif sev == "CRÍTICO":
            st.error(f"🔴 **SEVERIDAD: CRÍTICA (Peligro de Fallo)** — {rec_action}")
        else: # FALLO INMINENTE
            st.error(f"⚫ **SEVERIDAD: FALLO INMINENTE (PARADA INMEDIATA)** — {rec_action}")
            
        # Mostrar causa si hay anomalía
        if pred["is_anomaly"] and pred.get("cause"):
            st.info(f"🔍 **Análisis de Causa Raíz:** Se detecta una desviación inusual en: **{pred['cause']}**")

        # D. Grid de variables críticas
        st.markdown("### 📊 Sensores Críticos de la Turbina")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Temp LPC (S4)", f"{data.get('sensor_4', 0.0):.2f} K")
        col2.metric("Pres LPC (S11)", f"{data.get('sensor_11', 0.0):.2f} psia")
        col3.metric("Bypass Ratio (S15)", f"{data.get('sensor_15', 0.0):.4f}")
        col4.metric("Bleed LPT (S21)", f"{data.get('sensor_21', 0.0):.4f} lbm/s")

        # E. Expander de todas las variables
        with st.expander("🔍 Ver todos los 21 sensores y 3 ajustes C-MAPSS en tiempo real"):
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            
            # Diccionario de metadatos de sensores para mostrar nombres y unidades
            sensor_metadata = {
                "sensor_1": {"name": "Temp. Entrada Fan", "unit": "K"},
                "sensor_2": {"name": "Temp. Salida LPC", "unit": "K"},
                "sensor_3": {"name": "Temp. Salida HPC", "unit": "K"},
                "sensor_4": {"name": "Temp. Salida LPT", "unit": "K"},
                "sensor_5": {"name": "Pres. Entrada Fan", "unit": "psia"},
                "sensor_6": {"name": "Pres. Bypass Ducto", "unit": "psia"},
                "sensor_7": {"name": "Pres. Total Salida HPC", "unit": "psia"},
                "sensor_8": {"name": "Velocidad Física Fan", "unit": "rpm"},
                "sensor_9": {"name": "Velocidad Física Núcleo", "unit": "rpm"},
                "sensor_10": {"name": "Relación Presión Motor", "unit": ""},
                "sensor_11": {"name": "Pres. Estática Salida HPC", "unit": "psia"},
                "sensor_12": {"name": "Flujo Combustible a Ps30", "unit": "pps/psi"},
                "sensor_13": {"name": "Velocidad Corregida Fan", "unit": "rpm"},
                "sensor_14": {"name": "Velocidad Corregida Núcleo", "unit": "rpm"},
                "sensor_15": {"name": "Relación de Bypass", "unit": ""},
                "sensor_16": {"name": "Eficiencia Cámara Combustión", "unit": ""},
                "sensor_17": {"name": "Entalpía de Purga", "unit": ""},
                "sensor_18": {"name": "Velocidad Demandada Fan", "unit": "rpm"},
                "sensor_19": {"name": "Velocidad Corr. Demandada Fan", "unit": "rpm"},
                "sensor_20": {"name": "Purga Refrigeración HPT", "unit": "lbm/s"},
                "sensor_21": {"name": "Purga Refrigeración LPT", "unit": "lbm/s"},
            }
            
            # Ajustes
            col_s1.write("**Ajustes:**")
            col_s1.write(f"- Ajuste 1 (Altitud): {data.get('ajuste_1', 0.0):.2f}")
            col_s1.write(f"- Ajuste 2 (Mach): {data.get('ajuste_2', 0.0):.2f}")
            col_s1.write(f"- Ajuste 3 (Condición): {data.get('ajuste_3', 0.0):.2f}")
            
            # Sensores 1-7
            col_s2.write("**Sensores 1-7:**")
            for i in range(1, 8):
                meta = sensor_metadata[f"sensor_{i}"]
                unit_str = f" {meta['unit']}" if meta["unit"] else ""
                col_s2.write(f"- **Sensor {i}** ({meta['name']}): {data.get(f'sensor_{i}', 0.0):.2f}{unit_str}")
                
            # Sensores 8-14
            col_s3.write("**Sensores 8-14:**")
            for i in range(8, 15):
                meta = sensor_metadata[f"sensor_{i}"]
                unit_str = f" {meta['unit']}" if meta["unit"] else ""
                col_s3.write(f"- **Sensor {i}** ({meta['name']}): {data.get(f'sensor_{i}', 0.0):.2f}{unit_str}")
                
            # Sensores 15-21
            col_s4.write("**Sensores 15-21:**")
            for i in range(15, 22):
                meta = sensor_metadata[f"sensor_{i}"]
                unit_str = f" {meta['unit']}" if meta["unit"] else ""
                col_s4.write(f"- **Sensor {i}** ({meta['name']}): {data.get(f'sensor_{i}', 0.0):.2f}{unit_str}")

        st.markdown("---")

        # F. Diagnósticos IA y RUL
        col_diag1, col_diag2 = st.columns(2)
        
        # Color dinámico para el porcentaje gigante de salud
        health = pred["health_score"]
        h_color = "#10b981" if health >= 90 else "#eab308" if health >= 70 else "#f97316" if health >= 50 else "#ef4444" if health >= 30 else "#f43f5e"
        rul = pred["rul_estimated"]
        
        with col_diag1:
            st.subheader("Índice de Salud (IA)")
            st.markdown(f"<h1 style='text-align: center; color: {h_color}; font-size: 80px; margin-bottom: 10px;'>{health:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: #9ca3af;'>Estado del Compresor: <strong>{pred['status']} ({sev})</strong></p>", unsafe_allow_html=True)
            
        with col_diag2:
            st.subheader("Vida Útil Estimada (RUL)")
            st.markdown(f"<h1 style='text-align: center; color: #ffffff; font-size: 80px; margin-bottom: 10px;'>{rul:.1f} <span style='font-size: 24px; color: #9ca3af;'>ciclos</span></h1>", unsafe_allow_html=True)
            # RUL ProgressBar normalizado (asumiendo 150 ciclos como vida máxima típica)
            prog_val = max(0.0, min(1.0, rul / 150.0))
            st.progress(prog_val)
            st.markdown(f"<p style='text-align: center; color: #9ca3af;'>Ciclos estimados de operación segura</p>", unsafe_allow_html=True)

        st.markdown("---")

        # G. Gráficas históricas persistentes de InfluxDB
        st.subheader("📈 Historial de Diagnóstico y Degradación")
        
        res_hist = requests.get(f"{API_BASE}/predictions/history/{m_id}", timeout=2)
        if res_hist.status_code == 200:
            hist_payload = res_hist.json()
            history = hist_payload.get("history", [])
            
            if len(history) > 1:
                # Convertir a dataframe para graficar
                df_hist = pd.DataFrame(history)
                df_hist = df_hist.set_index("timestamp")
                
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.write("**Evolución del Score de Salud (%)**")
                    st.line_chart(df_hist["health_score"], color="#10b981")
                with col_g2:
                    st.write("**Evolución de la Vida Útil Restante (RUL)**")
                    st.line_chart(df_hist["rul_estimated"], color="#3b82f6")
            else:
                st.info("Recopilando datos históricos en InfluxDB. Las gráficas se mostrarán en breve...")
        else:
            st.warning("No se pudo obtener el historial de predicciones de InfluxDB.")

    except Exception as e:
        st.error(f"🚫 Error de comunicación con la API: {e}")

render_motor_details(motor_id)
