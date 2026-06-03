# =============================================================================
# 03_XAI_Validacion.py
# Validación, Explicabilidad (XAI) y Análisis de Señales
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import json
import builtins
warnings.filterwarnings('ignore')

# Permitir backend de archivos local en MLflow
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


# TensorFlow / Keras para cargar Autoencoder y LSTM
import tensorflow as tf
from tensorflow.keras.models import load_model

# Scikit-Learn metrics
from sklearn.metrics import mean_absolute_error

# XGBoost y SHAP para Explicabilidad
import xgboost as xgb
import shap

# Configuración visual
sns.set_theme(style="whitegrid")
os.makedirs('graficos', exist_ok=True)

# --- COMPROBACIÓN Y CONFIGURACIÓN DE GPU ---
print("\n--- COMPROBACIÓN DE GPU ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"TensorFlow ha detectado GPU(s): {gpus}")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
else:
    print("TensorFlow se ejecutará en CPU (para usar GPU en Windows se requiere WSL2 o DirectML compatible).")


# Cargamos el dataset de Test limpio
TEST_DATA_PATH = 'NASA_C-MAPSS/test_FD001_procesado.csv'
if not os.path.exists(TEST_DATA_PATH):
    raise FileNotFoundError(f"No se encuentra el dataset de test procesado en {TEST_DATA_PATH}. Ejecuta primero 01_EDA_Preprocesamiento.py")

df_test = pd.read_csv(TEST_DATA_PATH)
print(f"Dataset de test cargado. Forma: {df_test.shape}")

# Re-generación / Asegurar las 8 pistas (vibración y tendencia)
sensores_para_features = ['sensor_4', 'sensor_11', 'sensor_15', 'sensor_21']
for s in sensores_para_features:
    df_test[f'{s}_std'] = df_test.groupby('id_motor')[s].transform(lambda x: x.rolling(window=10).std())
    df_test[f'{s}_diff'] = df_test.groupby('id_motor')[s].diff(periods=5)
df_test.fillna(0, inplace=True)

# Lógica de secuenciación
def preparar_ventanas_3d(data, window_size=30):
    X = []
    y = []
    sensores = [col for col in data.columns if 'sensor' in col]
    for motor_id in data['id_motor'].unique():
        df_motor = data[data['id_motor'] == motor_id]
        if len(df_motor) > window_size:
            datos_sensores = df_motor[sensores].values
            if 'RUL' in df_motor.columns:
                datos_rul = df_motor['RUL'].values
            else:
                datos_rul = np.zeros(len(df_motor))
                
            for i in range(window_size, len(df_motor)):
                X.append(datos_sensores[i-window_size:i, :])
                y.append(datos_rul[i])
    return np.array(X), np.array(y)

# Aplicamos la ventana deslizante
X_test_3D, y_test_RUL = preparar_ventanas_3d(df_test, window_size=30)

# Aplanamos para XGBoost
n_muestras_test = X_test_3D.shape[0]
X_test_2D = X_test_3D.reshape(n_muestras_test, -1)

print(f"Secuencias 3D para Redes Neuronales preparadas: {X_test_3D.shape}")
print(f"Secuencias 2D para XGBoost preparadas: {X_test_2D.shape}")

# Carga de Modelos
carpeta_produccion = "./modelos_produccion"
if not os.path.exists(carpeta_produccion):
    raise FileNotFoundError(f"No se encuentra la carpeta {carpeta_produccion}. Ejecuta primero 02_Modelos_IA.py")

print("Cargando modelos entrenados...")
autoencoder = load_model(os.path.join(carpeta_produccion, "autoencoder_anomalias.keras"))

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(carpeta_produccion, "xgboost_clasificacion.json"))

lstm_model = load_model(os.path.join(carpeta_produccion, "lstm_regresion_rul.keras"))
print("Modelos cargados con éxito.")

# =============================================================================
# FASE 2: Explicabilidad Algorítmica (XAI) con SHAP
# =============================================================================
print("\n--- FASE 2: EXPLICABILIDAD ALGORÍTMICA (SHAP) ---")
original_float = builtins.float

class PatchedFloat(original_float):
    def __new__(cls, x=0):
        if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
            x = x.strip('[]')
        return original_float.__new__(cls, x)

builtins.float = PatchedFloat

try:
    print("Calculando valores SHAP...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_2D)
    builtins.float = original_float

    if isinstance(shap_values, list):
        shap_values_plot = shap_values[1]
    else:
        shap_values_plot = shap_values

    print("Valores SHAP calculados.")

    # Generación de nombres de columnas para la ventana 3D aplanada
    sensores_nombres = [col for col in df_test.columns if 'sensor' in col]
    nombres_columnas_660 = []
    # 30 ciclos x sensores
    for t in range(30, 0, -1):
        for sensor in sensores_nombres:
            nombres_columnas_660.append(f"{sensor}_t-{t}")

    # Ajustar a la longitud correcta (si difiere de 660, se adapta dinámicamente)
    n_features_expected = X_test_2D.shape[1]
    if len(nombres_columnas_660) != n_features_expected:
        nombres_columnas_660 = [f"feature_{i}" for i in range(n_features_expected)]

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values_plot, 
        X_test_2D, 
        feature_names=nombres_columnas_660, 
        max_display=20, 
        show=False
    )
    plt.title("Explicabilidad XAI: Impacto de las variables en la Alerta de Fallo", fontsize=16, pad=25)
    plt.tight_layout()
    plt.savefig('graficos/03_explicabilidad_shap.png')
    plt.close()
    print("Gráfico SHAP guardado en: graficos/03_explicabilidad_shap.png")

except Exception as e:
    builtins.float = original_float
    print(f"Error al calcular valores SHAP: {e}")

# =============================================================================
# FASE 3: Análisis de Señales Híbrido (R + ggplot2 con fallback Python)
# =============================================================================
print("\n--- FASE 3: AUDITORÍA DE SEÑALES ---")
try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    
    print("Ejecutando análisis híbrido con R (rpy2)...")
    robjects.globalenv['df_test'] = df_test
    
    robjects.r('''
        library(ggplot2)
        library(dplyr)
        
        motor_ejemplo <- df_test %>% filter(id_motor == 1)
        
        p1 <- ggplot(motor_ejemplo, aes(x=ciclo, y=sensor_4)) +
          geom_line(color="red", linewidth=1) +
          geom_smooth(method="loess", formula = 'y ~ x', color="black", linetype="dashed", se=FALSE) +
          labs(title="Auditoría de Señales: Sensor 4 (Motor ID: 1)", 
               subtitle="Tendencia de degradación ascendente detectada",
               x="Ciclos de Vida", y="Valor Normalizado") +
          theme_minimal()
        
        p2 <- ggplot(motor_ejemplo, aes(x=ciclo, y=sensor_3)) +
          geom_line(color="blue", linewidth=1) +
          geom_smooth(method="loess", formula = 'y ~ x', color="black", linetype="dashed", se=FALSE) +
          labs(title="Auditoría de Señales: Sensor 3 (Motor ID: 1)", 
               x="Ciclos de Vida", y="Valor Normalizado") +
          theme_minimal()
        
        dir.create("graficos", showWarnings = FALSE)
        ggsave("graficos/03_senal_sensor_4.png", plot = p1, width = 8, height = 4)
        ggsave("graficos/03_senal_sensor_3.png", plot = p2, width = 8, height = 4)
    ''')
    print("Gráficos de R creados con éxito y guardados en graficos/03_senal_sensor_4.png y graficos/03_senal_sensor_3.png")

except Exception as e:
    print(f"No se pudo ejecutar el análisis híbrido con R/rpy2: {e}")
    print("Como alternativa, generamos las mismas gráficas usando Python (Seaborn)...")
    try:
        motor_ejemplo_py = df_test[df_test['id_motor'] == 1]
        
        # Plot Sensor 4
        plt.figure(figsize=(8, 4))
        sns.lineplot(data=motor_ejemplo_py, x='ciclo', y='sensor_4', color='red', label='Sensor 4')
        # Regresión local aproximada con lowess de statsmodels si está disponible, si no regresión lineal estándar
        try:
            import statsmodels.api as sm
            lowess = sm.nonparametric.lowess
            z = lowess(motor_ejemplo_py['sensor_4'], motor_ejemplo_py['ciclo'], frac=0.3)
            plt.plot(z[:, 0], z[:, 1], color='black', linestyle='--', label='Tendencia (LOESS)')
        except ImportError:
            # Fallback a regplot simple si statsmodels no está
            sns.regplot(data=motor_ejemplo_py, x='ciclo', y='sensor_4', scatter=False, color='black', line_kws={"linestyle": "--"}, label='Tendencia')
            
        plt.title("Auditoría de Señales: Sensor 4 (Motor ID: 1) [Python]")
        plt.xlabel("Ciclos de Vida")
        plt.ylabel("Valor Normalizado")
        plt.legend()
        plt.tight_layout()
        plt.savefig('graficos/03_senal_sensor_4.png')
        plt.close()
        
        # Plot Sensor 3
        plt.figure(figsize=(8, 4))
        sns.lineplot(data=motor_ejemplo_py, x='ciclo', y='sensor_3', color='blue', label='Sensor 3')
        try:
            import statsmodels.api as sm
            lowess = sm.nonparametric.lowess
            z = lowess(motor_ejemplo_py['sensor_3'], motor_ejemplo_py['ciclo'], frac=0.3)
            plt.plot(z[:, 0], z[:, 1], color='black', linestyle='--', label='Tendencia (LOESS)')
        except ImportError:
            sns.regplot(data=motor_ejemplo_py, x='ciclo', y='sensor_3', scatter=False, color='black', line_kws={"linestyle": "--"}, label='Tendencia')
            
        plt.title("Auditoría de Señales: Sensor 3 (Motor ID: 1) [Python]")
        plt.xlabel("Ciclos de Vida")
        plt.ylabel("Valor Normalizado")
        plt.legend()
        plt.tight_layout()
        plt.savefig('graficos/03_senal_sensor_3.png')
        plt.close()
        
        print("Gráficos alternativos de Python guardados con éxito en la carpeta graficos/.")
    except Exception as py_err:
        print(f"Error al generar gráficos alternativos en Python: {py_err}")

# =============================================================================
# FASE 4: Pipeline Maestro de Inferencia (Función Unificada)
# =============================================================================
print("\n--- FASE 4: PIPELINE MAESTRO DE INFERENCIA ---")
def pipeline_inferencia_maestro(ventana_3d):
    """
    Recibe una ventana de (1, 30, n_sensores) y devuelve el diagnóstico completo.
    """
    # Aseguramos que los datos sean tensores de float32 para Keras
    ventana_tensor = tf.convert_to_tensor(ventana_3d, dtype=tf.float32)
    
    # 1. Autoencoder: Error de reconstrucción
    reconstruccion = autoencoder(ventana_tensor, training=False)
    mse = np.mean(np.power(ventana_3d - reconstruccion.numpy(), 2))
    
    # 2. XGBoost: Probabilidad de fallo (clasificación)
    ventana_2d = ventana_3d.reshape(1, -1)
    prob_critico = xgb_model.predict_proba(ventana_2d)[0][1]
    estado = "CRÍTICO" if prob_critico > 0.5 else "OPERATIVO"
    
    # 3. LSTM: Predicción de RUL
    rul_predicho_tensor = lstm_model(ventana_tensor, training=False)
    rul_predicho = rul_predicho_tensor.numpy()[0][0]
    
    return {
        "Estado": estado,
        "Confianza_Riesgo": f"{prob_critico:.2%}",
        "RUL_Estimado": int(round(rul_predicho)),
        "Anomalia_MSE": f"{mse:.6f}"
    }

print("Pipeline maestro de inferencia definido.")

# =============================================================================
# FASE 5: Validación Final contra el Ground Truth (RUL_FD001)
# =============================================================================
print("\n--- FASE 5: VALIDACIÓN CONTRA EL GROUND TRUTH ---")
ruta_rul_real = 'NASA_C-MAPSS/RUL_FD001.txt' 
if not os.path.exists(ruta_rul_real):
    print(f"Advertencia: No se encontró el archivo de ground truth en {ruta_rul_real}. No se realizará la validación final.")
else:
    # Cargar y clipear RUL a 115 ciclos
    y_true_raw = pd.read_csv(ruta_rul_real, header=None).values.flatten()
    y_true = np.clip(y_true_raw, None, 115)

    # Identificar la última ventana para cada motor
    indices_finales = []
    acumulado = 0
    for motor_id in df_test['id_motor'].unique():
        n_filas = len(df_test[df_test['id_motor'] == motor_id])
        n_ventanas = n_filas - 30 
        
        if n_ventanas > 0:
            indices_finales.append(acumulado + n_ventanas - 1)
            acumulado += n_ventanas

    print(f"Calculando precisión sobre {len(indices_finales)} motores detectados...")

    y_pred = []
    for idx in indices_finales:
        if idx < len(X_test_3D):
            ventana = X_test_3D[idx:idx+1] 
            resultado = pipeline_inferencia_maestro(ventana)
            y_pred.append(resultado['RUL_Estimado'])

    # Sincronizamos las longitudes
    y_true_final = y_true[:len(y_pred)]
    df_comparativa = pd.DataFrame({
        'Motor_ID': range(1, len(y_pred) + 1),
        'RUL_Real': y_true_final,
        'RUL_Predicho': y_pred
    })

    print("\n--- COMPARATIVA REAL VS PREDICCIÓN (ÚLTIMO CICLO) ---")
    print(df_comparativa.head(10).to_string(index=False))

    mae_final = mean_absolute_error(y_true_final, y_pred)
    print(f"\nMAE Final: {mae_final:.2f} ciclos")

    # Gráfico de Dispersión Final
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true_final, y_pred, alpha=0.5, c='blue', label='Predicciones')
    plt.plot([0, 115], [0, 115], color='red', linestyle='--', label='Predicción Perfecta')
    plt.title('Validación Final: RUL Real vs RUL Predicho')
    plt.xlabel('RUL Real (Capped 115)')
    plt.ylabel('RUL Predicho')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('graficos/03_validacion_final.png')
    plt.close()
    print("Gráfico de validación final guardado en: graficos/03_validacion_final.png")

print("\n[OK] Script 03_XAI_Validacion.py completado con éxito.")




