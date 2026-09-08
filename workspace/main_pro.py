import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def generate_benchmark_plots(df_res):
    """Genera y guarda un gráfico comparativo atractivo para publicar en LinkedIn"""
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Benchmark Performance: Scikit-Learn vs PySpark (HIGGS Dataset)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    models = df_res["Model"].unique()
    x = np.arange(len(models))
    width = 0.35

    # 1. Gráfico de Tiempos de Entrenamiento
    sk_train_times = df_res[df_res["Framework"] == "Scikit-Learn"][
        "Train Time (s)"
    ].values
    sp_train_times = df_res[df_res["Framework"] == "PySpark"]["Train Time (s)"].values

    rects1 = ax1.bar(
        x - width / 2, sk_train_times, width, label="Scikit-Learn", color="#2b5c8f"
    )
    rects2 = ax1.bar(
        x + width / 2, sp_train_times, width, label="PySpark", color="#e05a47"
    )

    ax1.set_ylabel("Tiempo de Entrenamiento (segundos)", fontsize=11, fontweight="bold")
    ax1.set_title("Tiempo de Entrenamiento por Modelo", fontsize=13, pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.legend(frameon=True, facecolor="white", framealpha=0.9)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Añadir valores sobre las barras
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax1.annotate(
            f"{height:.2f}s",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # 2. Gráfico de ROC-AUC
    sk_auc = df_res[df_res["Framework"] == "Scikit-Learn"]["ROC-AUC"].values
    sp_auc = df_res[df_res["Framework"] == "PySpark"]["ROC-AUC"].values

    rects3 = ax2.bar(
        x - width / 2, sk_auc, width, label="Scikit-Learn", color="#2b5c8f"
    )
    rects4 = ax2.bar(x + width / 2, sp_auc, width, label="PySpark", color="#e05a47")

    ax2.set_ylabel("ROC-AUC Score", fontsize=11, fontweight="bold")
    ax2.set_title("Precisión (ROC-AUC) por Modelo", fontsize=13, pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=11)
    ax2.set_ylim([0.5, 1.0])
    ax2.legend(frameon=True, facecolor="white", framealpha=0.9)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # Añadir valores sobre las barras
    for rect in rects3 + rects4:
        height = rect.get_height()
        ax2.annotate(
            f"{height:.4f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    output_img_path = "benchmark_results.png"
    plt.savefig(output_img_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(
        f"\n[GRAPH] Gráfico comparativo guardado exitosamente en: '{output_img_path}'"
    )


def main():
    print("==========================================================")
    print("   BENCHMARK: SCIKIT-LEARN vs PYSPARK (HIGGS DATASET)     ")
    print("==========================================================")

    DATA_PATH = "data/HIGGS.csv"
    N_SAMPLES = 1_000_000  # Puedes ajustar este número según tus pruebas
    RANDOM_STATE = 42

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"No se encontró el archivo en la ruta '{DATA_PATH}'. Asegúrate de que existe."
        )

    # Definir nombres de columnas: en HIGGS original la primera columna es 'label' y le siguen 28 características
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
    schema = StructType(
        [StructField("label", DoubleType(), True)]
        + [StructField(f"f{i}", DoubleType(), True) for i in range(28)]
    )

    df_spark = spark.read.csv(DATA_PATH, schema=schema, header=False).limit(N_SAMPLES)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_spark_vec = assembler.transform(df_spark).select("features", "label")

    train_sp, test_sp = df_spark_vec.randomSplit([0.8, 0.2], seed=RANDOM_STATE)
    train_sp.cache()
    train_sp.count()  # Forzar evaluación en memoria
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
    preds_lr_sp.count()  # Trigger action
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
    preds_rf_sp.count()  # Trigger action
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
    # 4. RESULTADOS FINALES Y GRÁFICO
    # ----------------------------------------------------
    print("\n[4/4] RESUMEN DE RESULTADOS")
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))

    # Generar y guardar imagen PNG
    generate_benchmark_plots(df_res)


if __name__ == "__main__":
    main()
