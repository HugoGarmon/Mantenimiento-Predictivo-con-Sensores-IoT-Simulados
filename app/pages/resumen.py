import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta

import os

# Configurar API URL
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_BASE = f"{API_URL}/api/v1"

st.markdown("# 🏢 Sala de Control - Estado de la Flota")
st.markdown("Monitorización consolidada en tiempo real de todos los motores IoT activos en la planta.")

# Inicializar sesión para selección de motor
if 'selected_motor_id' not in st.session_state:
    st.session_state.selected_motor_id = 1

# CSS para las tarjetas premium de los motores
st.markdown("""
<style>
    .metric-card {
        background-color: #1f2937;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #374151;
    }
    .motor-grid-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        border-left: 5px solid #10b981;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

@st.fragment(run_every=timedelta(seconds=2))
def render_fleet_dashboard():
    try:
        # 1. Consultar API
        res = requests.get(f"{API_BASE}/fleet/status", timeout=2)
        if res.status_code != 200:
            st.error("Error al obtener el estado de la flota desde la API.")
            return
            
        fleet_data = res.json()
        motors = fleet_data.get("motors", [])
        
        if not motors:
            st.warning("⚠️ No hay motores activos detectados. Esperando a que el simulador envíe telemetría...")
            return
            
        # 2. Renderizar KPIs Consolidados
        severity_counts = fleet_data.get("severity_counts", {})
        
        col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
        
        with col_kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #9ca3af; margin-bottom: 0px; font-size: 14px;">Motores Activos</p>
                <h2 style="color: #3b82f6; margin-top: 5px; margin-bottom: 5px;">{fleet_data.get("fleet_size", 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col_kpi2:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #10b981;">
                <p style="color: #9ca3af; margin-bottom: 0px; font-size: 14px;">Salud Óptima</p>
                <h2 style="color: #10b981; margin-top: 5px; margin-bottom: 5px;">{severity_counts.get("ÓPTIMO", 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col_kpi3:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #eab308;">
                <p style="color: #9ca3af; margin-bottom: 0px; font-size: 14px;">En Atención</p>
                <h2 style="color: #eab308; margin-top: 5px; margin-bottom: 5px;">{severity_counts.get("ATENCIÓN", 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col_kpi4:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #f97316;">
                <p style="color: #9ca3af; margin-bottom: 0px; font-size: 14px;">En Alerta</p>
                <h2 style="color: #f97316; margin-top: 5px; margin-bottom: 5px;">{severity_counts.get("ALERTA", 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col_kpi5:
            critical_total = severity_counts.get("CRÍTICO", 0) + severity_counts.get("FALLO INMINENTE", 0)
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #ef4444;">
                <p style="color: #9ca3af; margin-bottom: 0px; font-size: 14px;">Crítico / Parada</p>
                <h2 style="color: #ef4444; margin-top: 5px; margin-bottom: 5px;">{critical_total}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # 3. Grid Visual de Motores
        st.subheader("🖥️ Matriz de Operación de Motores")
        
        # Mapeo de colores por severidad
        color_map = {
            "ÓPTIMO": ("#10b981", "#064e3b", "#ecfdf5"),
            "ATENCIÓN": ("#eab308", "#713f12", "#fef9c3"),
            "ALERTA": ("#f97316", "#7c2d12", "#ffedd5"),
            "CRÍTICO": ("#ef4444", "#7f1d1d", "#fee2e2"),
            "FALLO INMINENTE": ("#f43f5e", "#4c0519", "#ffe4e6")
        }
        
        # Organizar en columnas para el grid (ej. 4 motores por fila)
        grid_cols = st.columns(4)
        
        for idx, motor in enumerate(motors):
            col = grid_cols[idx % 4]
            m_id = motor["id_motor"]
            health = motor["health_score"]
            rul = motor["rul_estimated"]
            sev = motor["severity_level"]
            
            border_color, bg_badge, fg_badge = color_map.get(sev, ("#9ca3af", "#374151", "#ffffff"))
            
            with col:
                st.markdown(f"""
                <div class="motor-grid-card" style="border-left-color: {border_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: bold; font-size: 16px; color: #f3f4f6;">Motor {m_id:02d}</span>
                        <span style="background-color: {bg_badge}; color: {border_color}; border: 1px solid {border_color}; padding: 1px 6px; border-radius: 4px; font-size: 10px; font-weight: bold;">
                            {sev}
                        </span>
                    </div>
                    <div style="margin-top: 10px; display: flex; justify-content: space-between; align-items: baseline;">
                        <span style="font-size: 26px; font-weight: bold; color: {border_color};">{health:.1f}%</span>
                        <span style="font-size: 12px; color: #9ca3af;">RUL: {rul:.1f} ciclos</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Botón nativo de streamlit debajo de cada tarjeta para seleccionarlo
                if st.button(f"Ver Motor {m_id:02d}", key=f"select_{m_id}"):
                    st.session_state.selected_motor_id = m_id
                    st.switch_page("pages/motor_individual.py")
                    
        st.markdown("---")
        
        # 4. Tabla de Clasificación de Riesgo e Incidencias
        col_risk, col_events = st.columns([1, 1])
        
        with col_risk:
            st.subheader("⚠️ Ranking de Riesgo (Mantenimiento Urgente)")
            df_motors = pd.DataFrame(motors)
            # Ordenar por menor RUL y salud
            df_motors = df_motors.sort_values(by=["rul_estimated", "health_score"]).reset_index(drop=True)
            df_motors = df_motors.rename(columns={
                "id_motor": "ID Motor",
                "health_score": "Salud (IA)",
                "rul_estimated": "RUL Est. (ciclos)",
                "severity_level": "Severidad",
                "last_updated": "Último Reporte"
            })
            
            # Dar formato a la salud y RUL basándonos en si existen esas columnas
            if "Salud (IA)" in df_motors.columns:
                df_motors["Salud (IA)"] = df_motors["Salud (IA)"].map(lambda x: f"{float(x):.1f}%")
            if "RUL Est. (ciclos)" in df_motors.columns:
                df_motors["RUL Est. (ciclos)"] = df_motors["RUL Est. (ciclos)"].map(lambda x: f"{float(x):.1f} ciclos")
            
            # Mostrar solo las columnas formateadas
            cols_to_show = ["ID Motor", "Salud (IA)", "RUL Est. (ciclos)", "Severidad", "Último Reporte"]
            st.dataframe(df_motors[cols_to_show].head(5), use_container_width=True)
            
        with col_events:
            st.subheader("📋 Registro Automatizado de Alarmas")
            # Filtrar motores con severidad mayor a Óptimo
            alerts = [m for m in motors if m["severity_level"] != "ÓPTIMO"]
            if alerts:
                alert_records = []
                for a in alerts:
                    reportado_str = "N/A"
                    if a.get("last_updated"):
                        try:
                            reportado_str = pd.to_datetime(a["last_updated"]).strftime("%H:%M:%S")
                        except Exception:
                            reportado_str = str(a["last_updated"])
                    alert_records.append({
                        "Motor": f"Motor {a['id_motor']:02d}",
                        "Severidad": a["severity_level"],
                        "Salud": f"{a['health_score']:.1f}%",
                        "RUL": f"{a['rul_estimated']:.1f} ciclos",
                        "Reportado": reportado_str
                    })
                st.table(pd.DataFrame(alert_records))
            else:
                st.success("✅ No hay incidencias activas en este momento.")

    except Exception as e:
        st.error(f"🚫 Error de conexión con el Backend de IA: {e}")
        st.info(f"Asegúrate de que la API de FastAPI esté activa en {API_URL}")

render_fleet_dashboard()
