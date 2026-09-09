import json
import os
import time
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def run_sklearn_benchmark(DATA_PATH: str, N_SAMPLES=None):
    print("==========================================================")
    print(" 1. BENCHMARK: SCIKIT-LEARN ")
    print("==========================================================")

    RANDOM_STATE = 42

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"No se encontró el archivo '{DATA_PATH}'.")

    feature_cols = [f"f{i}" for i in range(28)]
    column_names = ["label"] + feature_cols

    print(f"Cargando dataset desde '{DATA_PATH}'...")
    if N_SAMPLES is not None:
        df_pandas = pd.read_csv(
            DATA_PATH, header=None, names=column_names, nrows=N_SAMPLES
        )
    else:
        df_pandas = pd.read_csv(DATA_PATH, header=None, names=column_names)

    X_train, X_test, y_train, y_test = train_test_split(
        df_pandas[feature_cols],
        df_pandas["label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    results = []

    # Logistic Regression
    print("  -> Entrenando Logistic Regression...")
    t0 = time.time()
    lr_sk = LogisticRegression(max_iter=100, random_state=RANDOM_STATE)
    lr_sk.fit(X_train, y_train)
    t_train_lr = time.time() - t0

    t0 = time.time()
    preds_lr = lr_sk.predict_proba(X_test)[:, 1]
    t_pred_lr = time.time() - t0
    auc_lr = roc_auc_score(y_test, preds_lr)

    results.append(
        {
            "Framework": "Scikit-Learn",
            "Model": "Logistic Regression",
            "Train Time (s)": round(t_train_lr, 3),
            "Predict Time (s)": round(t_pred_lr, 3),
            "ROC-AUC": round(auc_lr, 4),
        }
    )

    # Random Forest
    print("  -> Entrenando Random Forest (50 árboles, depth=10)...")
    t0 = time.time()
    rf_sk = RandomForestClassifier(
        n_estimators=50, max_depth=10, n_jobs=-1, random_state=RANDOM_STATE
    )
    rf_sk.fit(X_train, y_train)
    t_train_rf = time.time() - t0

    t0 = time.time()
    preds_rf = rf_sk.predict_proba(X_test)[:, 1]
    t_pred_rf = time.time() - t0
    auc_rf = roc_auc_score(y_test, preds_rf)

    results.append(
        {
            "Framework": "Scikit-Learn",
            "Model": "Random Forest",
            "Train Time (s)": round(t_train_rf, 3),
            "Predict Time (s)": round(t_pred_rf, 3),
            "ROC-AUC": round(auc_rf, 4),
        }
    )

    # Guardar resultados
    os.makedirs(
        os.path.join(os.path.dirname(__file__), "data", "results"), exist_ok=True
    )
    output_json = os.path.join(
        os.path.dirname(__file__),
        "data",
        "results",
        f"results_sklearn.json",
    )
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n[OK] Resultados de Scikit-Learn guardados en '{output_json}'.")


if __name__ == "__main__":
    # run_sklearn_benchmark(
    #     DATA_PATH=os.path.join(os.path.dirname(__file__), "data", "raw", "HIGGS.csv"),
    #     N_SAMPLES=1_000_000,
    # )
    run_sklearn_benchmark(
        DATA_PATH=os.path.join(os.path.dirname(__file__), "data", "raw", "HIGGS.csv")
    )
