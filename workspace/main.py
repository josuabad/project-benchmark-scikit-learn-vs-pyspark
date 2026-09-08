import time
import os
import numpy as np
import pandas as pd

# Algoritmos y Métricas - Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Algoritmos y Métricas - PySpark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import (
    RandomForestClassifier as SparkRF,
    LogisticRegression as SparkLR,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def main():
    print("==========================================================")
    print("   BENCHMARK: SCIKIT-LEARN vs PYSPARK (HIGGS DATASET)     ")
    print("==========================================================")

    DATA_PATH = "data/HIGGS.csv"
    N_SAMPLES = 1_000_000  # Cambia este valor para probar más/menos filas
    RANDOM_STATE = 42

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"No se encontró el archivo en la ruta '{DATA_PATH}'. Asegúrate de que existe."
        )

    # En el dataset HIGGS la primera columna es la etiqueta (0/1) y le siguen 28 características
    feature_cols = [f"f{i}" for i in range(28)]
    column_names = ["label"] + feature_cols

    print(f"\n[1/4] Cargando dataset desde '{DATA_PATH}'...")
    print(f"      Cargando las primeras {N_SAMPLES:,} filas...")

    # Cargar datos desde CSV local usando Pandas
    df_pandas = pd.read_csv(DATA_PATH, header=None, names=column_names, nrows=N_SAMPLES)

    results = []

    # ----------------------------------------------------
    # 2. BENCHMARK SCIKIT-LEARN
    # ----------------------------------------------------
    print("\n[2/4] Ejecutando pruebas en Scikit-Learn (n_jobs=-1)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df_pandas[feature_cols],
        df_pandas["label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # Logistic Regression Sklearn
    print("  -> Entrenando Regresión Logística...")
    t0 = time.time()
    lr_sk = LogisticRegression(max_iter=100, n_jobs=-1, random_state=RANDOM_STATE)
    lr_sk.fit(X_train, y_train)
    t_train_lr_sk = time.time() - t0

    t0 = time.time()
    preds_lr_sk = lr_sk.predict_proba(X_test)[:, 1]
    t_pred_lr_sk = time.time() - t0
    auc_lr_sk = roc_auc_score(y_test, preds_lr_sk)

    results.append(
        {
            "Framework": "Scikit-Learn",
            "Model": "Logistic Regression",
            "Train Time (s)": round(t_train_lr_sk, 3),
            "Predict Time (s)": round(t_pred_lr_sk, 3),
            "ROC-AUC": round(auc_lr_sk, 4),
        }
    )

    # Random Forest Sklearn
    print("  -> Entrenando Random Forest (50 árboles, depth=10)...")
    t0 = time.time()
    rf_sk = RandomForestClassifier(
        n_estimators=50, max_depth=10, n_jobs=-1, random_state=RANDOM_STATE
    )
    rf_sk.fit(X_train, y_train)
    t_train_rf_sk = time.time() - t0

    t0 = time.time()
    preds_rf_sk = rf_sk.predict_proba(X_test)[:, 1]
    t_pred_rf_sk = time.time() - t0
    auc_rf_sk = roc_auc_score(y_test, preds_rf_sk)

    results.append(
        {
            "Framework": "Scikit-Learn",
            "Model": "Random Forest",
            "Train Time (s)": round(t_train_rf_sk, 3),
            "Predict Time (s)": round(t_pred_rf_sk, 3),
            "ROC-AUC": round(auc_rf_sk, 4),
        }
    )

    # ----------------------------------------------------
    # 3. BENCHMARK PYSPARK
    # ----------------------------------------------------
    print("\n[3/4] Inicializando SparkSession local (8 Cores, 10GB RAM)...")
    spark = (
        SparkSession.builder.appName("Benchmark_HIGGS")
        .config("spark.driver.memory", "10g")
        .config("spark.executor.memory", "10g")
        .config("spark.master", "local[8]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    # Cargar datos desde el CSV local en PySpark
    t0 = time.time()

    # Esquema explícito
    schema = StructType(
        [StructField("label", DoubleType(), True)]
        + [StructField(f"f{i}", DoubleType(), True) for i in range(28)]
    )

    df_spark = spark.read.csv(DATA_PATH, schema=schema, header=False).limit(N_SAMPLES)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_spark_vec = assembler.transform(df_spark).select("features", "label")

    train_sp, test_sp = df_spark_vec.randomSplit([0.8, 0.2], seed=RANDOM_STATE)
    train_sp.cache()
    train_sp.count()  # Forzar ejecución en memoria
    t_prep_sp = time.time() - t0
    print(
        f"      Tiempo de preparación Spark (Carga CSV + Assembler + Cache): {t_prep_sp:.3f}s"
    )

    evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )

    # Logistic Regression PySpark
    print("  -> Entrenando Logistic Regression en PySpark...")
    t0 = time.time()
    lr_sp = SparkLR(maxIter=100, regParam=0.01)
    lr_sp_model = lr_sp.fit(train_sp)
    t_train_lr_sp = time.time() - t0

    t0 = time.time()
    preds_lr_sp = lr_sp_model.transform(test_sp)
    preds_lr_sp.count()  # Action trigger
    t_pred_lr_sp = time.time() - t0
    auc_lr_sp = evaluator.evaluate(preds_lr_sp)

    results.append(
        {
            "Framework": "PySpark",
            "Model": "Logistic Regression",
            "Train Time (s)": round(t_train_lr_sp, 3),
            "Predict Time (s)": round(t_pred_lr_sp, 3),
            "ROC-AUC": round(auc_lr_sp, 4),
        }
    )

    # Random Forest PySpark
    print("  -> Entrenando Random Forest en PySpark...")
    t0 = time.time()
    rf_sp = SparkRF(numTrees=50, maxDepth=10, seed=RANDOM_STATE)
    rf_sp_model = rf_sp.fit(train_sp)
    t_train_rf_sp = time.time() - t0

    t0 = time.time()
    preds_rf_sp = rf_sp_model.transform(test_sp)
    preds_rf_sp.count()  # Action trigger
    t_pred_rf_sp = time.time() - t0
    auc_rf_sp = evaluator.evaluate(preds_rf_sp)

    results.append(
        {
            "Framework": "PySpark",
            "Model": "Random Forest",
            "Train Time (s)": round(t_train_rf_sp, 3),
            "Predict Time (s)": round(t_pred_rf_sp, 3),
            "ROC-AUC": round(auc_rf_sp, 4),
        }
    )

    spark.stop()

    # ----------------------------------------------------
    # 4. RESULTADOS FINALES
    # ----------------------------------------------------
    print("\n[4/4] RESUMEN DE RESULTADOS")
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()
