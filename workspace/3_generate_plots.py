import glob
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_benchmark_plots(df_res):
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Benchmark Performance for 11_000_000 rows: Scikit-Learn vs PySpark",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    models = df_res["Model"].unique()
    x = np.arange(len(models))
    width = 0.35

    # 1. Tiempos de Entrenamiento
    sk_train = df_res[df_res["Framework"] == "Scikit-Learn"]["Train Time (s)"].values
    sp_train = df_res[df_res["Framework"] == "PySpark"]["Train Time (s)"].values

    rects1 = ax1.bar(
        x - width / 2, sk_train, width, label="Scikit-Learn", color="#2b5c8f"
    )
    rects2 = ax1.bar(x + width / 2, sp_train, width, label="PySpark", color="#e05a47")

    ax1.set_ylabel("Tiempo de Entrenamiento (segundos)", fontsize=11, fontweight="bold")
    ax1.set_title("Tiempo de Entrenamiento por Modelo", fontsize=13, pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.legend(frameon=True, facecolor="white", framealpha=0.9)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax1.annotate(
            f"{height:.2f}s",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # 2. ROC-AUC
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

    output_dir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(output_dir, exist_ok=True)
    output_img_path = os.path.join(output_dir, "benchmark_results.png")

    plt.savefig(output_img_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(
        f"\n[GRAPH] Gráfico comparativo guardado exitosamente en: '{output_img_path}'"
    )


def main():
    print("==========================================================")
    print(" 3. GENERANDO GRÁFICAS DE RESULTADOS ")
    print("==========================================================")

    json_files = glob.glob(
        os.path.join(os.path.dirname(__file__), "data", "results", "results_*.json")
    )
    if not json_files:
        raise FileNotFoundError(
            "No se encontraron archivos JSON en la carpeta 'data/'."
        )

    all_results = []
    for file_path in json_files:
        with open(file_path, "r") as f:
            all_results.extend(json.load(f))

    df_res = pd.DataFrame(all_results)
    print("\nRESUMEN DE RESULTADOS COMBINADOS:")
    print(df_res.to_string(index=False))

    generate_benchmark_plots(df_res)


if __name__ == "__main__":
    main()
