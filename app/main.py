import streamlit as st

st.set_page_config(
    page_title="Industrial Monitor 4.0 - Mantenimiento Predictivo",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definir las páginas usando st.Page
resumen_page = st.Page("pages/resumen.py", title="Resumen de Flota", icon="🏢", default=True)
motor_page = st.Page("pages/motor_individual.py", title="Detalle de Motor", icon="⚙️")
historico_page = st.Page("pages/analisis_historico.py", title="Análisis Histórico", icon="📊")
config_page = st.Page("pages/configuracion.py", title="Configuración del Sistema", icon="🛠️")

# Cargar la navegación
pg = st.navigation({
    "Operaciones": [resumen_page, motor_page],
    "Análisis e Infraestructura": [historico_page, config_page]
})

# Ejecutar la página seleccionada
pg.run()