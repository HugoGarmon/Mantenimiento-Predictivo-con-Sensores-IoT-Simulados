FROM python:3.11-slim

WORKDIR /app

# Copiar e instalar requerimientos de Python dedicados del dashboard
COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código del Dashboard y configuración de tema en el lugar correcto
COPY app/ ./app/
COPY app/.streamlit/ ./.streamlit/

# Exponer el puerto de Streamlit
EXPOSE 8501

# Comando para arrancar el Dashboard en modo headless
CMD ["streamlit", "run", "app/main.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
