# =============================================================================
# 01_EDA_Preprocesamiento.py
# Análisis Exploratorio de Datos (EDA) y Preprocesamiento
# Dataset: NASA C-MAPSS (train_FD001.txt / test_FD001.txt)
# =============================================================================
# Transforma los datos crudos de sensores en datasets limpios y normalizados
# para entrenar modelos de mantenimiento predictivo (Autoencoder, LSTM, XGBoost).
#
# Pipeline:
#   1. Carga del dataset y cálculo del RUL
#   2. EDA: distribuciones, correlaciones, tendencias
#   3. Limpieza: eliminación de sensores constantes
#   4. Suavizado: media móvil de 5 ciclos
#   5. Multicolinealidad: eliminación de sensores redundantes (corr > 0.98)
#   6. Ingeniería de características: volatilidad (std) y tendencia (diff)
#   7. Normalización Min-Max
#   8. Exportación de train y test procesados
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# ---- Configuración de rutas ----
DATA_DIR = 'NASA_C-MAPSS'
TRAIN_FILE = f'{DATA_DIR}/train_FD001.txt'
TEST_FILE = f'{DATA_DIR}/test_FD001.txt'
TRAIN_OUTPUT = f'{DATA_DIR}/train_FD001_procesado.csv'
TEST_OUTPUT = f'{DATA_DIR}/test_FD001_procesado.csv'

# =============================================================================
# FASE 1: Carga del Dataset y Cálculo del RUL
# =============================================================================

# Columnas del dataset C-MAPSS
nombres_columnas = ['id_motor', 'ciclo', 'ajuste_1', 'ajuste_2', 'ajuste_3'] + \
                   [f'sensor_{i}' for i in range(1, 22)]

# Cargar dataset de entrenamiento
df = pd.read_csv(TRAIN_FILE, sep=r'\s+', header=None, names=nombres_columnas)

# Calcular RUL (Remaining Useful Life) = ciclos restantes hasta el fallo
ciclos_maximos = df.groupby('id_motor')['ciclo'].transform('max')
df['RUL'] = ciclos_maximos - df['ciclo']

print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Motores únicos: {df['id_motor'].nunique()}")

# =============================================================================
# FASE 2: Análisis Exploratorio (EDA)
# =============================================================================

# Verificar valores nulos
print(f"\nValores nulos por columna:\n{df.isna().sum().sum()} total")

# Distribución del RUL
plt.figure(figsize=(6, 4))
sns.histplot(df['RUL'], bins=50, kde=True)
plt.title("Distribución del RUL")
plt.xlabel("RUL (ciclos restantes)")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig(f'{DATA_DIR}/eda_distribucion_rul.png', dpi=150)
plt.close()

# Evolución del Sensor 11 en Motor 1 (sensor representativo)
motor_1 = df[df['id_motor'] == 1]
plt.figure(figsize=(10, 4))
plt.plot(motor_1['ciclo'], motor_1['sensor_11'])
plt.title("Evolución del Sensor 11 en el Motor 1")
plt.xlabel("Ciclo")
plt.ylabel("Valor del sensor")
plt.tight_layout()
plt.savefig(f'{DATA_DIR}/eda_sensor11_motor1.png', dpi=150)
plt.close()

# Correlación de sensores con el RUL
correlacion_rul = df.corr(numeric_only=True)['RUL'].drop(
    ['id_motor', 'ciclo', 'RUL']
).sort_values()

print("\nTOP 3 Sensores con correlación NEGATIVA al RUL (suben al degradarse):")
print(correlacion_rul.head(3))
print("\nTOP 3 Sensores con correlación POSITIVA al RUL (bajan al degradarse):")
print(correlacion_rul.tail(3))

# Matriz de correlación
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', annot=False)
plt.title('Matriz de Correlación - NASA C-MAPSS (Motor FD001)', fontsize=16)
plt.tight_layout()
plt.savefig(f'{DATA_DIR}/eda_matriz_correlacion.png', dpi=150)
plt.close()

print("\nGráficos EDA guardados en carpeta NASA_C-MAPSS/")

# =============================================================================
# FASE 3: Limpieza - Eliminación de Sensores Constantes (Varianza Cero)
# =============================================================================

# Sensores con varianza nula no aportan información predictiva
sensores_constantes = [col for col in df.columns if df[col].std() == 0]
print(f"\nSensores constantes eliminados ({len(sensores_constantes)}): {sensores_constantes}")

df_limpio = df.drop(columns=sensores_constantes)

# Columnas de sensores restantes (para transformaciones posteriores)
columnas_sensores = [col for col in df_limpio.columns if 'sensor' in col]

# =============================================================================
# FASE 4: Suavizado - Media Móvil de 5 Ciclos
# =============================================================================
# Reduce el ruido de alta frecuencia manteniendo la tendencia de degradación

df_limpio[columnas_sensores] = (
    df_limpio.groupby('id_motor')[columnas_sensores]
    .rolling(window=5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# Visualización antes/después del suavizado (Sensor 11, Motor 1)
plt.figure(figsize=(12, 4))
plt.plot(df[df['id_motor'] == 1]['ciclo'],
         df[df['id_motor'] == 1]['sensor_11'],
         color='red', alpha=0.5, label='Original (Ruido)')
plt.plot(df_limpio[df_limpio['id_motor'] == 1]['ciclo'],
         df_limpio[df_limpio['id_motor'] == 1]['sensor_11'],
         color='blue', linewidth=2, label='Suavizado')
plt.title('Efecto de la Media Móvil en sensor_11')
plt.legend()
plt.tight_layout()
plt.savefig(f'{DATA_DIR}/eda_suavizado_sensor11.png', dpi=150)
plt.close()

# =============================================================================
# FASE 5: Eliminación de Multicolinealidad (Correlación > 0.98)
# =============================================================================

corr_matrix = df_limpio.corr(numeric_only=True).abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]

print(f"Sensores redundantes eliminados (corr > 0.98): {to_drop}")

df_final = df_limpio.drop(columns=to_drop)

# Scatter plot de correlación perfecta (Sensor 9 vs 14) como ejemplo
plt.figure(figsize=(6, 6))
plt.scatter(df['sensor_9'], df['sensor_14'], alpha=0.1, color='purple')
plt.title('Correlación perfecta entre Sensor 9 y Sensor 14', fontsize=14)
plt.xlabel('Valor del Sensor 9')
plt.ylabel('Valor del Sensor 14')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f'{DATA_DIR}/eda_correlacion_sensor9_vs_14.png', dpi=150)
plt.close()

# =============================================================================
# FASE 6: Ingeniería de Características (Volatilidad y Tendencia)
# =============================================================================
# Para los sensores más críticos:
#   - std (ventana 10): captura la volatilidad/vibración de la señal
#   - diff (5 ciclos): captura la velocidad de degradación

sensores_para_features = ['sensor_4', 'sensor_11', 'sensor_15', 'sensor_21']

for s in sensores_para_features:
    df_final[f'{s}_std'] = df_final.groupby('id_motor')[s].transform(
        lambda x: x.rolling(window=10).std()
    )
    df_final[f'{s}_diff'] = df_final.groupby('id_motor')[s].diff(periods=5)

df_final.fillna(0, inplace=True)

print(f"\nFeatures añadidas: {len(sensores_para_features) * 2} "
      f"({[f'{s}_std, {s}_diff' for s in sensores_para_features]})")

# =============================================================================
# FASE 7: Normalización Min-Max [0, 1]
# =============================================================================

scaler = MinMaxScaler()
columnas_a_normalizar = [col for col in df_final.columns
                         if 'sensor' in col or 'ajuste' in col]

df_final[columnas_a_normalizar] = scaler.fit_transform(df_final[columnas_a_normalizar])

print(f"Columnas normalizadas: {len(columnas_a_normalizar)}")

# Guardar Scaler para producción
os.makedirs('modelos_produccion', exist_ok=True)
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'modelos_produccion/scaler.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("Scaler guardado en modelos_produccion/scaler.pkl y models/scaler.pkl")

# =============================================================================
# FASE 8: Exportación del Dataset de Entrenamiento
# =============================================================================

df_final.to_csv(TRAIN_OUTPUT, index=False)
print(f"\n[TRAIN] Dataset procesado guardado: {TRAIN_OUTPUT}")
print(f"  Shape: {df_final.shape}")

# =============================================================================
# FASE 9: Pipeline de Inferencia - Preprocesamiento del Test
# =============================================================================
# Se aplican EXACTAMENTE las mismas transformaciones que al train.
# Se usa scaler.transform() (NO fit_transform) para evitar data leakage.

df_test = pd.read_csv(TEST_FILE, sep=r'\s+', header=None, names=nombres_columnas)

# Eliminar sensores constantes (mismos que en train)
df_test_limpio = df_test.drop(columns=sensores_constantes)

# Media móvil de 5 ciclos
df_test_limpio[columnas_sensores] = (
    df_test_limpio.groupby('id_motor')[columnas_sensores]
    .rolling(window=10, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# Eliminar sensores redundantes (mismos que en train)
df_test_final = df_test_limpio.drop(columns=to_drop)

# Ingeniería de características (mismas features que en train)
for s in sensores_para_features:
    df_test_final[f'{s}_std'] = df_test_final.groupby('id_motor')[s].transform(
        lambda x: x.rolling(window=10).std()
    )
    df_test_final[f'{s}_diff'] = df_test_final.groupby('id_motor')[s].diff(periods=5)

df_test_final.fillna(0, inplace=True)

# Normalización con el MISMO scaler del train (transform, no fit_transform)
columnas_actualizadas = [col for col in df_test_final.columns
                         if 'sensor' in col or 'ajuste' in col]
df_test_final[columnas_actualizadas] = scaler.transform(df_test_final[columnas_actualizadas])

# Exportar
df_test_final.to_csv(TEST_OUTPUT, index=False)
print(f"\n[TEST] Dataset procesado guardado: {TEST_OUTPUT}")
print(f"  Shape: {df_test_final.shape}")
print("\n[OK] Pipeline de preprocesamiento completado.")
