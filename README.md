# Project: Benchmark Scikit-learn vs PySpark

Benchmark Scikit-learn vs PySpark entrenando modelos supervisados de ML

## Pasos para usar el Dockerfile:

1. **Guardar el archivo:** Guarda el contenido anterior en un archivo llamado `Dockerfile` (sin extensión) o renombra el archivo descargado.
2. **Construir la imagen:**

```bash
docker build -t jupyter-spark:latest .

```

3. **Ejecutar el contenedor (standalone):**

```bash
docker run -p 8888:8888 -p 4040:4040 -v $(pwd):/home/jovyan/work jupyter-spark:latest

```

4. **En caso de reintento:**

```bash
docker compose build --no-cache

```

## Descargar los datos

Please, execute the following command to download the necessary data files for this project:

```bash
mkdir -p data/raw && cd data/
wget https://archive.ics.uci.edu/static/public/280/higgs.zip -O raw/higgs.zip
unzip data/raw/higgs.zip -d data/
gunzip data/HIGGS.csv.gz
rm -rf data/raw/ # OPCIONAL: Remove the zip file after extraction
```
