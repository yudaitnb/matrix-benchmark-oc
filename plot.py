import matplotlib.pyplot as plt
import csv


def load_data(csv_file):
    sizes, naive_times, numpy_times = [], [], []
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            N = int(row["N"])
            naive_time = float(row["NaiveTime"]) if row["NaiveTime"] else None
            numpy_time = float(row["NumpyTime"]) if row["NumpyTime"] else None
            sizes.append(N)
            naive_times.append(naive_time)
            numpy_times.append(numpy_time)
    return sizes, naive_times, numpy_times


def plot_graph(sizes, naive_times, numpy_times, N_limit=None, which="both", filename="plot.png"):
    fig, ax = plt.subplots(figsize=(10, 6))

    # 範囲フィルタ
    filtered = [(s, nt, yt) for s, nt, yt in zip(sizes, naive_times, numpy_times)
                if (N_limit is None or s <= N_limit)]
    fsizes = [s for s, _, _ in filtered]
    fnaive = [nt for _, nt, _ in filtered]
    fnumpy = [yt for _, _, yt in filtered]

    if which in ("naive", "both"):
        s_naive = [s for s, t in zip(fsizes, fnaive) if t is not None]
        t_naive = [t for t in fnaive if t is not None]
        ax.plot(s_naive, t_naive, marker='o', label="Naïve", color="blue")

    if which in ("numpy", "both"):
        s_numpy = [s for s, t in zip(fsizes, fnumpy) if t is not None]
        t_numpy = [t for t in fnumpy if t is not None]
        ax.plot(s_numpy, t_numpy, marker='s', label="NumPy", color="green")

    ax.set_xlabel("Matrix Size N")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title(f"Matrix Multiplication Benchmark ({which}, N≤{N_limit if N_limit else 'ALL'})")
    ax.grid(True)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"✅ {filename} を保存しました。")


if __name__ == "__main__":
    sizes, naive_times, numpy_times = load_data("results.csv")

    # Nの制限と表示対象の組み合わせ (6通り)
    configurations = [
        (400, "naive", "naive_upto_400.png"),
        (400, "numpy", "numpy_upto_400.png"),
        (400, "both",  "both_upto_400.png"),
        (2000, "naive", "naive_upto_2000.png"),
        (2000, "numpy", "numpy_upto_2000.png"),
        (2000, "both",  "both_upto_2000.png"),
    ]

    for N_limit, mode, filename in configurations:
        plot_graph(sizes, naive_times, numpy_times, N_limit, mode, filename)
