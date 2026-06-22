# =============================================================================
# 02_Modelos_IA.py
# Modelado, Entrenamiento y Comparativa de Algoritmos
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil 
import warnings
warnings.filterwarnings('ignore')

# Permitir backend de archivos local en MLflow
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (LSTM, GRU, Conv1D, MaxPooling1D, Flatten,
                                     Dense, Input, Dropout, RepeatVector, TimeDistributed)
from tensorflow.keras.optimizers import Adam

# XGBoost y MLflow
import xgboost as xgb
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature  

# Configuración de estilo de plots
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

# Habilitar GPU en XGBoost si está disponible
import subprocess
try:
    subprocess.check_output('nvidia-smi', stderr=subprocess.DEVNULL)
    xgb_device = 'cuda'
    print("XGBoost detectó GPU NVIDIA. Configurando para usar GPU ('cuda').")
except Exception:
    xgb_device = 'cpu'
    print("XGBoost no detectó GPU compatible. Configurando para usar CPU.")


# Cargamos el dataset limpio y normalizado
DATA_PATH = 'NASA_C-MAPSS/train_FD001_procesado.csv'
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"No se encuentra el dataset preprocesado en {DATA_PATH}. Ejecuta primero 01_EDA_Preprocesamiento.py")

df = pd.read_csv(DATA_PATH)
print(f"Dataset cargado con éxito. Forma: {df.shape}")

# =============================================================================
# FASE 1: Secuenciación Temporal (Sliding Window)
# =============================================================================
def preparar_ventanas_3d(data, window_size=30):
    X = []
    y = []
    
    sensores = [col for col in data.columns if 'sensor' in col]
    
    for motor_id in data['id_motor'].unique():
        df_motor = data[data['id_motor'] == motor_id]
        
        if len(df_motor) > window_size:
            datos_sensores = df_motor[sensores].values
            datos_rul = df_motor['RUL'].values
            
            for i in range(window_size, len(df_motor)):
                X.append(datos_sensores[i-window_size:i, :])
                y.append(datos_rul[i])
                
    return np.array(X), np.array(y)

# Aplicamos la transformación (ventana de 30 ciclos)
X_3D, y_RUL = preparar_ventanas_3d(df, window_size=30)
print(f"Forma de las secuencias de entrada (X_3D): {X_3D.shape}")
print(f"Forma de las etiquetas de RUL (y_RUL): {y_RUL.shape}")

# =============================================================================
# FASE 2: Detección de Anomalías (Comparativa de Modelos)
# =============================================================================
print("\n--- FASE 2: DETECCIÓN DE ANOMALÍAS ---")
# Filtramos solo los datos "sanos" (RUL > 65) para el entrenamiento no supervisado
sanos_idx = y_RUL > 65
X_sanos_3D = X_3D[sanos_idx]

# Dividimos en entrenamiento y validación para el Autoencoder
X_train_ae, X_val_ae = train_test_split(X_sanos_3D, test_size=0.2, random_state=42)
print(f"Hay {X_train_ae.shape[0]} ventanas de motores sanos.")

# Preparación de datos 2D para Isolation Forest y OCSVM (no admiten 3D)
n_muestras_train = X_train_ae.shape[0]
n_muestras_total = X_3D.shape[0]

X_train_2D = X_train_ae.reshape(n_muestras_train, -1)
X_total_2D = X_3D.reshape(n_muestras_total, -1)

# --- 2.1 Modelo 1: Autoencoder (Keras) ---
n_sensores = X_3D.shape[2]
window_size = X_3D.shape[1]

input_layer = Input(shape=(window_size, n_sensores))
encoder = LSTM(16, activation='relu', return_sequences=False)(input_layer)
encoder = Dense(8, activation='relu')(encoder)
decoder = RepeatVector(window_size)(encoder)
decoder = LSTM(16, activation='relu', return_sequences=True)(decoder)
output_layer = TimeDistributed(Dense(n_sensores))(decoder)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mae')

print("Entrenando Autoencoder (15 épocas)...")
history_ae = autoencoder.fit(
    X_train_ae, X_train_ae, 
    epochs=15, 
    batch_size=64, 
    validation_data=(X_val_ae, X_val_ae), 
    verbose=1
)

# --- 2.2 Modelos 2 y 3: Isolation Forest y One-Class SVM ---
print("Entrenando Isolation Forest...")
iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
iso_forest.fit(X_train_2D)

print("Entrenando One-Class SVM (sobre muestra del 30% para eficiencia)...")
np.random.seed(42)
sample_indices = np.random.choice(X_train_2D.shape[0], int(X_train_2D.shape[0] * 0.3), replace=False)
X_train_sample = X_train_2D[sample_indices]

oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
oc_svm.fit(X_train_sample)

# --- 2.3 Evaluación y Comparativa ---
print("Calculando anomalías con todos los modelos...")
X_3D_reconstruido = autoencoder.predict(X_3D, verbose=0)
mae_ae = np.mean(np.abs(X_3D_reconstruido - X_3D), axis=(1, 2))

preds_if = iso_forest.predict(X_total_2D)
anomalias_if = (preds_if == -1).astype(int)

# OCSVM (subconjunto para rapidez)
indices_eval = np.linspace(0, len(X_total_2D)-1, 5000, dtype=int)
X_eval_ocsvm = X_total_2D[indices_eval]
y_RUL_eval = y_RUL[indices_eval]

preds_ocsvm = oc_svm.predict(X_eval_ocsvm)
anomalias_ocsvm = (preds_ocsvm == -1).astype(int)

# --- 2.4 Análisis Visual ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# Gráfico Autoencoder
ax1.scatter(y_RUL, mae_ae, alpha=0.1, color='orange', s=10)
z = np.polyfit(y_RUL, mae_ae, 3)
p = np.poly1d(z)
ax1.plot(np.sort(y_RUL)[::-1], p(np.sort(y_RUL)[::-1]), "r--", linewidth=2, label='Tendencia del Error')
ax1.invert_xaxis()
ax1.axvline(x=30, color='black', linestyle=':', label='Zona Crítica (RUL < 30)')
ax1.set_title('Autoencoder: Error de Reconstrucción vs RUL')
ax1.set_xlabel('RUL (Ciclos Restantes)')
ax1.set_ylabel('Error (MAE)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico Isolation Forest vs OCSVM
bins = [0, 30, 65, 130]
labels = ['Crítico (<30)', 'Degradación (30-65)', 'Sano (>65)']

df_if = pd.DataFrame({'RUL': y_RUL, 'Anomalia': anomalias_if})
df_if['Fase'] = pd.cut(df_if['RUL'], bins=bins, labels=labels)
porcentaje_if = df_if.groupby('Fase', observed=False)['Anomalia'].mean() * 100

df_ocsvm = pd.DataFrame({'RUL': y_RUL_eval, 'Anomalia': anomalias_ocsvm})
df_ocsvm['Fase'] = pd.cut(df_ocsvm['RUL'], bins=bins, labels=labels)
porcentaje_ocsvm = df_ocsvm.groupby('Fase', observed=False)['Anomalia'].mean() * 100

x = np.arange(len(labels))
width = 0.35

ax2.bar(x - width/2, porcentaje_if, width, label='Isolation Forest', color='skyblue')
ax2.bar(x + width/2, porcentaje_ocsvm, width, label='One-Class SVM', color='lightgreen')
ax2.set_ylabel('% de Secuencias detectadas como Anomalía')
ax2.set_title('Detección Clásica: % de Anomalías por Fase Operativa')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('graficos/02_deteccion_anomalias.png')
plt.close()
print("Gráfico de anomalías guardado en: graficos/02_deteccion_anomalias.png")

# =============================================================================
# FASE 3: Clasificación Binaria (Comparativa de Modelos Supervisados)
# =============================================================================
print("\n--- FASE 3: CLASIFICACIÓN BINARIA ---")
# Creación de etiquetas binarias (1 si RUL <= 30, 0 en caso contrario)
y_class = (y_RUL <= 30).astype(int)

# Aplanamiento de la matriz 3D para modelos tabulares
X_2D = X_3D.reshape(X_3D.shape[0], -1)

# División en conjuntos de entrenamiento (80%) y test (20%) stratified
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_2D, y_class, test_size=0.2, random_state=42, stratify=y_class
)

print(f"Datos preparados para clasificación.")
print(f"Dimensiones de entrenamiento: {X_train_c.shape}")
print(f"Distribución de clases (Sano vs Crítico): {np.bincount(y_train_c)}")

# 1. Entrenamiento de Random Forest
print("Entrenando Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    class_weight='balanced',
    random_state=42, 
    n_jobs=-1
)
rf_model.fit(X_train_c, y_train_c)

# 2. Entrenamiento de XGBoost
print("Entrenando XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=6, 
    scale_pos_weight=4.68,
    eval_metric='logloss', 
    random_state=42,
    device=xgb_device
)
xgb_model.fit(X_train_c, y_train_c)

# 3. Entrenamiento del Perceptrón Multicapa (MLP)
print("Entrenando Red Neuronal (MLP)...")
mlp_model = Sequential([
    Input(shape=(X_train_c.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_mlp = mlp_model.fit(
    X_train_c, y_train_c,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    verbose=0 
)

# Predicciones
preds_rf = rf_model.predict(X_test_c)
preds_xgb = xgb_model.predict(X_test_c)
preds_mlp = (mlp_model.predict(X_test_c, verbose=0) > 0.5).astype(int).flatten()

# Graficar Matrices de Confusión
class_names = ['Sano (0)', 'Crítico (1)']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Comparativa de Matrices de Confusión', fontsize=16, fontweight='bold')

cm_rf = confusion_matrix(y_test_c, preds_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=class_names)
disp_rf.plot(ax=axes[0], cmap='Blues', values_format='d', colorbar=False)
axes[0].set_title('Random Forest')

cm_xgb = confusion_matrix(y_test_c, preds_xgb)
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=class_names)
disp_xgb.plot(ax=axes[1], cmap='Greens', values_format='d', colorbar=False)
axes[1].set_title('XGBoost')

cm_mlp = confusion_matrix(y_test_c, preds_mlp)
disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=class_names)
disp_mlp.plot(ax=axes[2], cmap='Oranges', values_format='d', colorbar=False)
axes[2].set_title('Perceptrón Multicapa (MLP)')

plt.tight_layout()
plt.savefig('graficos/02_matrices_confusion.png')
plt.close()
print("Matrices de confusión guardadas en: graficos/02_matrices_confusion.png")

# Reporte XGBoost
print("-" * 55)
print("REPORTE DE CLASIFICACIÓN: XGBOOST (Modelo Seleccionado)")
print("-" * 55)
print(classification_report(y_test_c, preds_xgb, target_names=class_names))

# =============================================================================
# FASE 4: Predicción de Vida Útil Restante (Comparativa de Modelos de Regresión)
# =============================================================================
print("\n--- FASE 4: REGRESIÓN (PREDICCIÓN DE RUL) ---")
# División de los datos 3D en Train y Test (80/20) para regresión
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_3D, y_RUL, test_size=0.2, random_state=42
)

print(f"Dimensiones de entrenamiento (Regresión 3D): {X_train_r.shape}")

# Parámetros de entrenamiento
EPOCHS = 30
BATCH_SIZE = 64
input_shape = (X_train_r.shape[1], X_train_r.shape[2])

# 1. Arquitectura LSTM
print("Entrenando Modelo LSTM (30 épocas)...")
lstm_model = Sequential([
    Input(shape=input_shape), 
    LSTM(64, return_sequences=True), 
    Dropout(0.3),                    
    LSTM(32, return_sequences=False),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
history_lstm = lstm_model.fit(X_train_r, y_train_r, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=0)
print("LSTM entrenado.")

# 2. Arquitectura GRU
print("Entrenando Modelo GRU (30 épocas)...")
gru_model = Sequential([
    Input(shape=input_shape),
    GRU(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
history_gru = gru_model.fit(X_train_r, y_train_r, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=0)
print("GRU entrenado.")

# 3. Arquitectura CNN-1D
print("Entrenando Modelo CNN-1D (30 épocas)...")
cnn_model = Sequential([
    Input(shape=input_shape),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
history_cnn = cnn_model.fit(X_train_r, y_train_r, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=0)
print("CNN-1D entrenada.")

# Generamos predicciones continuas para los tres modelos
print("Generando predicciones de regresión...")
preds_lstm = lstm_model.predict(X_test_r, verbose=0).flatten()
preds_gru = gru_model.predict(X_test_r, verbose=0).flatten()
preds_cnn = cnn_model.predict(X_test_r, verbose=0).flatten()

def print_metrics(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} -> MAE: {mae:.2f} ciclos | RMSE: {rmse:.2f} | R²: {r2:.4f}")

print("-" * 55)
print("MÉTRICAS DE RENDIMIENTO (PREDICCIÓN DE RUL)")
print("-" * 55)
print_metrics("LSTM  ", y_test_r, preds_lstm)
print_metrics("GRU   ", y_test_r, preds_gru)
print_metrics("CNN-1D", y_test_r, preds_cnn)
print("-" * 55)

# Visualización: RUL Real vs Predicho
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Dispersión de Predicciones: RUL Real vs RUL Predicho', fontsize=16, fontweight='bold')

max_val = max(y_test_r.max(), 350)
ideal_line = [0, max_val]

# LSTM
axes[0].scatter(y_test_r, preds_lstm, alpha=0.3, color='blue', s=10)
axes[0].plot(ideal_line, ideal_line, color='red', linestyle='--')
axes[0].set_title('Modelo LSTM')
axes[0].set_xlabel('RUL Real (Ciclos)')
axes[0].set_ylabel('RUL Predicho (Ciclos)')

# GRU
axes[1].scatter(y_test_r, preds_gru, alpha=0.3, color='green', s=10)
axes[1].plot(ideal_line, ideal_line, color='red', linestyle='--')
axes[1].set_title('Modelo GRU')
axes[1].set_xlabel('RUL Real (Ciclos)')

# CNN-1D
axes[2].scatter(y_test_r, preds_cnn, alpha=0.3, color='orange', s=10)
axes[2].plot(ideal_line, ideal_line, color='red', linestyle='--')
axes[2].set_title('Modelo CNN-1D')
axes[2].set_xlabel('RUL Real (Ciclos)')

plt.tight_layout()
plt.savefig('graficos/02_comparativa_regresion.png')
plt.close()
print("Gráfico de regresión guardado en: graficos/02_comparativa_regresion.png")

# =============================================================================
# FASE 5: Gestión del Ciclo de Vida del Modelo (MLOps con MLflow)
# =============================================================================
print("\n--- FASE 5: MLOPS Y EXPORTACIÓN DE MODELOS ---")
ruta_mlruns = "./mlruns"
ruta_trash = os.path.join(ruta_mlruns, ".trash")

# Borramos si existe para resetear el tracking
if os.path.exists(ruta_mlruns):
    try:
        shutil.rmtree(ruta_mlruns)
    except Exception as e:
        print(f"Advertencia al borrar mlruns: {e}")

os.makedirs(ruta_mlruns, exist_ok=True)
os.makedirs(ruta_trash, exist_ok=True)

# Configuración del tracking (corregido para Windows usando slashes '/' y triple slash 'file:///').
mlflow.set_tracking_uri("file:///" + os.path.abspath(ruta_mlruns).replace('\\', '/'))
mlflow.set_experiment("Mantenimiento_Predictivo_Motores")

print("Iniciando el registro del modelo en MLflow...")
with mlflow.start_run(run_name="LSTM_RUL_Prediction"):
    # Parámetros
    mlflow.log_param("model_type", "LSTM")
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("lstm_units", 64)

    # Métricas
    lstm_mae = mean_absolute_error(y_test_r, preds_lstm)
    lstm_rmse = np.sqrt(mean_squared_error(y_test_r, preds_lstm))
    lstm_r2 = r2_score(y_test_r, preds_lstm)
    
    mlflow.log_metric("MAE", lstm_mae)
    mlflow.log_metric("RMSE", lstm_rmse)
    mlflow.log_metric("R2_Score", lstm_r2)

    # Firma e inferencia del modelo
    firma_modelo = infer_signature(X_test_r, preds_lstm)

    # Registro en MLflow
    mlflow.keras.log_model(
        model=lstm_model, 
        artifact_path="modelo_lstm_rul", 
        signature=firma_modelo
    )

print("Experimento registrado con éxito en MLflow.")
print(f"Métricas del modelo LSTM -> MAE: {lstm_mae:.2f} | RMSE: {lstm_rmse:.2f} | R²: {lstm_r2:.4f}")

# --- Anexo: Exportación Física de Modelos ---
carpeta_salida = "./models"
os.makedirs(carpeta_salida, exist_ok=True)
print(f"\nIniciando exportación física en: '{carpeta_salida}'...")

# 1. Autoencoder
ruta_ae = os.path.join(carpeta_salida, "autoencoder_anomalias.keras")
autoencoder.save(ruta_ae)
print(f"Modelo Autoencoder exportado en: {ruta_ae}")

# 2. XGBoost
ruta_xgb = os.path.join(carpeta_salida, "xgboost_clasificacion.json")
xgb_model.save_model(ruta_xgb)
print(f"Modelo XGBoost exportado en: {ruta_xgb}")

# 3. LSTM
ruta_lstm = os.path.join(carpeta_salida, "lstm_regresion_rul.keras")
lstm_model.save(ruta_lstm)
print(f"Modelo LSTM exportado en: {ruta_lstm}")

print("\n[OK] Script 02_Modelos_IA.py completado con éxito.")




