# Base Image: Python 3.14 Slim
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar OpenJDK 21 y dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-21-jre-headless procps curl && apt-get clean && rm -rf /var/lib/apt/lists/*

# Configurar variable de entorno apuntando a Java 21
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

WORKDIR /workspace

# Instalar JupyterLab, PySpark y librerías
RUN pip install --no-cache-dir jupyterlab pyspark findspark delta-spark

EXPOSE 8888 4040

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]