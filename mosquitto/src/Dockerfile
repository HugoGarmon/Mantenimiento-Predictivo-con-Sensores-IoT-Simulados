FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente y el dataset de test
COPY data_consumer.py .
COPY sensor_simulator.py .
COPY test_FD001.txt .

# Comando por defecto (será sobreescrito en compose.yml para cada servicio)
CMD ["python", "sensor_simulator.py"]
