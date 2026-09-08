import pyspark
from pyspark.sql import SparkSession

# Inicializar la sesión de Spark
spark = SparkSession.builder.appName("TestSpark").getOrCreate()

# 1. Verificar la versión de PySpark y del motor Spark
print("Versión de Spark/PySpark:", spark.version)

# 2. Prueba rápida del motor procesando datos
df = spark.createDataFrame([("Spark", 1), ("PySpark", 2)], ["Modulo", "ID"])
df.show()
