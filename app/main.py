import streamlit as st
import requests
import pandas as pd
import time

st.set_page_config(page_title="Monitor Predictivo 4.0", layout="wide")

st.title("🏭 Monitorización de Activos - Machine_01")

# Placeholder para que el dashboard se refresque dinámicamente
placeholder = st.empty()

# Historial para la gráfica
history = []

while True:
    try:
        # 1. Pedir predicción a nuestra API
        response = requests.get("http://127.0.0.1:8000/predict").json()
        
        data = response["sensor_data"]
        pred = response["prediction"]
        
        with placeholder.container():
            # Fila 1: Métricas de Sensores
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Temperatura", f"{data['temperatura']} °C")
            col2.metric("Vibración", f"{data['vibracion']} mm/s")
            col3.metric("Presión", f"{data['presion']} bar")
            col4.metric("RPM", f"{data['rpm']}")

            st.markdown("---")

            # Fila 2: Predicciones de IA
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Estado de Salud (IA)")
                color = "green" if pred['health_score'] > 75 else "orange" if pred['health_score'] > 40 else "red"
                st.markdown(f"<h1 style='text-align: center; color: {color};'>{pred['health_score']}%</h1>", unsafe_allow_html=True)
                st.info(f"Diagnóstico: **{pred['status']}**")

            with c2:
                st.subheader("Vida Útil Restante (RUL)")
                st.write(f"Estimación: **{pred['rul_estimated']} horas**")
                st.progress(pred['health_score'] / 100)

            # Gráfica de evolución
            history.append({"Health": pred['health_score'], "Time": time.strftime("%H:%M:%S")})
            if len(history) > 20: history.pop(0)
            st.line_chart(pd.DataFrame(history).set_index("Time"))

    except Exception as e:
        st.error(f"Esperando conexión con la API... {e}")
    
    time.sleep(2) # Actualizar cada 2 segundos