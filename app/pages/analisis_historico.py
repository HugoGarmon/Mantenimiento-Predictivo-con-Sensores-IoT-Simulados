import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import io

import os

# Configurar API URL
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_BASE = f"{API_URL}/api/v1"

st.markdown("# 📊 Análisis Histórico e Informes")
st.markdown("Compare el rendimiento de degradación entre motores de la flota y descargue los informes de operación.")

# 1. Obtener la lista de motores activos
active_motors = []
try:
    res_fleet = requests.get(f"{API_BASE}/fleet/status", timeout=2)
    if res_fleet.status_code == 200:
        active_motors = [m["id_motor"] for m in res_fleet.json().get("motors", [])]
except Exception:
    pass

if not active_motors:
    st.warning("⚠️ No se detectan motores activos con historial. Asegúrese de que el simulador y la API estén corriendo.")
else:
    # 2. Selección de motores a comparar
    selected_compare = st.multiselect(
        "Seleccione los motores que desea comparar:",
        active_motors,
        default=active_motors[:2] if len(active_motors) >= 2 else active_motors
    )
    
    if not selected_compare:
        st.info("Seleccione al menos un motor para ver la comparativa.")
    else:
        # 3. Consultar históricos y compilar DataFrames
        histories_health = {}
        histories_rul = {}
        all_records = []
        
        for m_id in selected_compare:
            try:
                res_hist = requests.get(f"{API_BASE}/predictions/history/{m_id}", timeout=2)
                if res_hist.status_code == 200:
                    history_list = res_hist.json().get("history", [])
                    if history_list:
                        # Rellenar diccionarios de comparativa
                        for h in history_list:
                            ts = h["timestamp"]
                            
                            if ts not in histories_health:
                                histories_health[ts] = {}
                            histories_health[ts][f"Motor {m_id:02d}"] = h["health_score"]
                            
                            if ts not in histories_rul:
                                histories_rul[ts] = {}
                            histories_rul[ts][f"Motor {m_id:02d}"] = h["rul_estimated"]
                            
                            # Consolidar para tabla y CSV
                            all_records.append({
                                "Timestamp": ts,
                                "Motor": f"Motor {m_id:02d}",
                                "Salud (Health Score)": h["health_score"],
                                "Vida Útil (RUL)": h["rul_estimated"],
                                "Severidad": h["severity_level"],
                                "Anomalía": "SÍ" if h["is_anomaly"] else "NO",
                                "Causa Probable": h["cause"] or "Ninguna"
                            })
            except Exception as e:
                st.error(f"Error cargando datos del motor {m_id:02d}: {e}")
                
        # 4. Renderizar gráficas comparativas
        if histories_health:
            df_compare_health = pd.DataFrame.from_dict(histories_health, orient="index").sort_index()
            df_compare_rul = pd.DataFrame.from_dict(histories_rul, orient="index").sort_index()
            
            st.markdown("### 📈 Comparativa de Desgaste en Tiempo Real")
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.write("**Curva Comparativa de Salud (%)**")
                st.line_chart(df_compare_health)
            with col_chart2:
                st.write("**Evolución de RUL (Ciclos)**")
                st.line_chart(df_compare_rul)
                
            st.markdown("---")
            
            # 5. Tabla de logs completa y descarga de informes CSV
            st.subheader("📋 Historial Consolidado de Predicciones")
            
            df_log = pd.DataFrame(all_records)
            st.dataframe(df_log, use_container_width=True)
            
            # Botón de exportación a CSV
            csv_buffer = io.StringIO()
            df_log.to_csv(csv_buffer, index=False)
            csv_str = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Descargar Historial en formato CSV",
                data=csv_str,
                file_name=f"reporte_mantenimiento_flota_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No hay suficientes datos históricos registrados en InfluxDB para los motores seleccionados.")
