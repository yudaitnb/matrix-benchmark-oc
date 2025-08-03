import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("result", exist_ok=True)

def load_dataframe(csv_file):
    return pd.read_csv(csv_file)

def plot_methods(df, methods, N_limit=None, title="Benchmark", filename="plot.png"):
    plt.figure(figsize=(10, 6))

    # フィルタ処理
    if N_limit:
        df = df[df["N"] <= N_limit]

    # 各手法のプロット
    for method in methods:
        if method in df.columns:
            plt.plot(df["N"], df[method], label=method, marker='o')

    plt.xlabel("Matrix Size N")
    plt.ylabel("Execution Time (seconds)")
    plt.title(f"{title} (N ≤ {N_limit if N_limit else 'ALL'})")
    plt.grid(True)
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    save_path = os.path.join("result", filename)
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}")

if __name__ == "__main__":
    df = load_dataframe(os.path.join("result", "results10.csv"))

    configs = [
        (400, ["Naive"], "Naive Only", "naive_upto_400.png"),
        (400, ["Naive", "NaiveC"], "Naive Comparison", "naive_c_upto_400.png"),
        (4000, ["Naive", "NaiveC", "BlockedC"], "Blocked C Comparison", "blocked_c_upto_400.png"),
        (4000, ["Naive", "NaiveC", "BlockedC", "BlockedOMP"], "Blocked OMP Comparison", "blocked_omp_upto_4000.png"),
        (4000, ["Naive", "NaiveC", "BlockedC", "BlockedOMP", "BlockedOMPTuning"], "Architecture-dependent Tuning Comparison", "optimized_omp_upto_4000.png"),
        (4000, ["Naive", "NaiveC", "BlockedC", "BlockedOMP", "BlockedOMPTuning", "Numpy"], "All Methods", "all_upto_4000.png"),
    ]

    for limit, methods, title, fname in configs:
        plot_methods(df, methods, N_limit=limit, title=title, filename=fname)
