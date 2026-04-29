import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime

st.set_page_config(page_title="Monitor Predictivo 4.0 - IA Real", layout="wide")

st.title("🏭 Monitorización Proactiva - Machine_01")

# --- LÓGICA DE PERSISTENCIA ---
if 'history_df' not in st.session_state:
    st.session_state.history_df = pd.DataFrame(columns=["Time", "Health", "RUL"])

if 'events' not in st.session_state:
    st.session_state.events = []

placeholder = st.empty()

while True:
    try:
        # 1. Petición a la API
        response = requests.get("http://127.0.0.1:8000/predict", timeout=2).json()
        
        if response.get("status") == "error":
            st.warning(f"Esperando datos: {response.get('message')}")
            time.sleep(2)
            continue

        data = response["sensor_data"]
        pred = response["prediction"]
        curr_time = datetime.now().strftime("%H:%M:%S")

        # 2. Actualizar Historial de Gráficas
        new_row = pd.DataFrame([{"Time": curr_time, "Health": pred['health_score'], "RUL": pred['rul_estimated']}])
        st.session_state.history_df = pd.concat([st.session_state.history_df, new_row], ignore_index=True)
        if len(st.session_state.history_df) > 30:
            st.session_state.history_df = st.session_state.history_df.iloc[1:]

        # 3. Gestión de Eventos (Solo si hay anomalía)
        if pred['is_anomaly']:
            new_event = {
                "Hora": curr_time,
                "Estado": pred['status'],
                "Causa": f"Código {pred['failure_code']}",
                "Salud": f"{pred['health_score']}%"
            }
            # Evitar duplicados en el mismo segundo
            if not st.session_state.events or st.session_state.events[0]['Hora'] != curr_time:
                st.session_state.events.insert(0, new_event)
                if len(st.session_state.events) > 10: st.session_state.events.pop()

        # --- 4. DIBUJAR INTERFAZ ---
        with placeholder.container():
            # ALERTAS
            if pred['is_anomaly']:
                st.error(f"🚨 ALERTAS ACTIVAS: {pred['status']} (Código de Fallo: {pred['failure_code']})")
            else:
                st.success("✅ SISTEMA OPERANDO EN PARÁMETROS NORMALES")

            # MÉTRICAS
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Temperatura", f"{data['temperatura']} °C")
            col2.metric("Vibración", f"{data['vibracion']} mm/s")
            col3.metric("Presión", f"{data['presion']} bar")
            col4.metric("RPM", f"{data['rpm']}")

            st.markdown("---")

            # IA Y RUL
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Estado de Salud (IA)")
                h_color = "green" if pred['health_score'] > 80 else "orange" if pred['health_score'] > 50 else "red"
                st.markdown(f"<h1 style='text-align: center; color: {h_color};'>{pred['health_score']}%</h1>", unsafe_allow_html=True)
                st.info(f"Diagnóstico: {pred['status']}")
            
            with c2:
                st.subheader("Vida Útil Estimada (RUL)")
                st.markdown(f"<h1 style='text-align: center; color: white;'>{pred['rul_estimated']} <span style='font-size: 20px;'>horas</span></h1>", unsafe_allow_html=True)
                # Normalizamos el progress: salud/100 para que no pete
                st.progress(max(0.0, min(1.0, pred['health_score'] / 100)))
            
            # GRÁFICAS
            st.markdown("### 📈 Análisis de Tendencias Temporales")
            df_display = st.session_state.history_df.set_index("Time")
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.write("**Evolución de Salud (%)**")
                st.line_chart(df_display["Health"], color="#2ecc71")
            with col_g2:
                st.write("**Tendencia de Vida Útil (Horas)**")
                st.line_chart(df_display["RUL"], color="#3498db")

            # TABLA DE INCIDENCIAS (Dentro del container para que se limpie al refrescar)
            st.markdown("### 📋 Registro de Incidencias Recientes")
            if st.session_state.events:
                st.table(pd.DataFrame(st.session_state.events))
            else:
                st.info("No se han detectado anomalías en la sesión actual.")

    except Exception as e:
        st.error(f"🚫 Error de conexión con el Backend: {e}")
    
    time.sleep(2)