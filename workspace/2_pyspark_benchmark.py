import json
import os
import time

from pyspark.ml.classification import LogisticRegression as SparkLR
from pyspark.ml.classification import RandomForestClassifier as SparkRF
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, StructField, StructType


def run_spark_benchmark(DATA_PATH: str, N_SAMPLES=None):
    print("==========================================================")
    print(" 2. BENCHMARK: PYSPARK ")
    print("==========================================================")

    RANDOM_STATE = 42

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"No se encontró el archivo '{DATA_PATH}'.")

    # SparkSession con límites de memoria adaptados (6GB)
    spark = (
        SparkSession.builder.appName("Benchmark_HIGGS_Spark")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.driver.memoryOverhead", "1g")
        .config("spark.master", "local[*]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    feature_cols = [f"f{i}" for i in range(28)]
    schema = StructType(
        [StructField("label", DoubleType(), True)]
        + [StructField(col, DoubleType(), True) for col in feature_cols]
    )

    t0 = time.time()
    if N_SAMPLES is not None:
        df_spark = spark.read.csv(DATA_PATH, schema=schema, header=False).limit(
            N_SAMPLES
        )
    else:
        df_spark = spark.read.csv(DATA_PATH, schema=schema, header=False)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_spark_vec = assembler.transform(df_spark).select("features", "label")

    train_sp, test_sp = df_spark_vec.randomSplit([0.8, 0.2], seed=RANDOM_STATE)
    train_sp.cache()
    train_sp.count()  # Materializar cache
    t_prep_sp = time.time() - t0
    print(f"Tiempo de preparación Spark: {t_prep_sp:.3f}s")

    evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )

    results = []

    # Logistic Regression
    print("  -> Entrenando Logistic Regression en PySpark...")
    t0 = time.time()
    lr_sp = SparkLR(maxIter=100, regParam=0.01)
    lr_sp_model = lr_sp.fit(train_sp)
    t_train_lr = time.time() - t0

    t0 = time.time()
    preds_lr = lr_sp_model.transform(test_sp)
    preds_lr.count()
    t_pred_lr = time.time() - t0
    auc_lr = evaluator.evaluate(preds_lr)

    results.append(
        {
            "Framework": "PySpark",
            "Model": "Logistic Regression",
            "Train Time (s)": round(t_train_lr, 3),
            "Predict Time (s)": round(t_pred_lr, 3),
            "ROC-AUC": round(auc_lr, 4),
        }
    )

    # Random Forest
    print("  -> Entrenando Random Forest en PySpark...")
    t0 = time.time()
    rf_sp = SparkRF(numTrees=50, maxDepth=10, seed=RANDOM_STATE)
    rf_sp_model = rf_sp.fit(train_sp)
    t_train_rf = time.time() - t0

    t0 = time.time()
    preds_rf = rf_sp_model.transform(test_sp)
    preds_rf.count()
    t_pred_rf = time.time() - t0
    auc_rf = evaluator.evaluate(preds_rf)

    results.append(
        {
            "Framework": "PySpark",
            "Model": "Random Forest",
            "Train Time (s)": round(t_train_rf, 3),
            "Predict Time (s)": round(t_pred_rf, 3),
            "ROC-AUC": round(auc_rf, 4),
        }
    )

    spark.stop()

    # Guardar resultados
    os.makedirs(
        os.path.join(os.path.dirname(__file__), "data", "results"), exist_ok=True
    )
    output_json = os.path.join(
        os.path.dirname(__file__),
        "data",
        "results",
        f"results_pyspark.json",
    )
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n[OK] Resultados de PySpark guardados en '{output_json}'.")


if __name__ == "__main__":
    # run_spark_benchmark(
    #     DATA_PATH=os.path.join(os.path.dirname(__file__), "data", "raw", "HIGGS.csv"),
    #     N_SAMPLES=1_000_000,
    # )
    run_spark_benchmark(
        DATA_PATH=os.path.join(os.path.dirname(__file__), "data", "raw", "HIGGS.csv")
    )
