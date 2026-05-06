import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

OUTPUTS_DIR = "outputs"
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")


def load_metrics() -> pd.DataFrame:
    rows = []
    for dir_name in os.listdir(OUTPUTS_DIR):
        metrics_path = os.path.join(OUTPUTS_DIR, dir_name, "metrics.json")
        if not os.path.isfile(metrics_path):
            continue
        with open(metrics_path) as f:
            data = json.load(f)
        rows.append({
            "dir": dir_name,
            "is_baseline": dir_name == "vanilla_kgt_rnn",
            "split": data["split"],
            "top_k": data["top_k"],
            "num_gold": data["num_gold"],
            "sym_count": data["sym_count"],
            "neu_count": data["neu_count"],
            "sym_hit": data["sym_hit"],
            "neu_hit": data["neu_hit"],
            "union_hit": data["union_hit"],
            "sym_coverage": data["sym_coverage"],
            "neu_coverage": data["neu_coverage"],
            "union_coverage": data["union_coverage"],
        })
    df = pd.DataFrame(rows).sort_values(["is_baseline", "split", "top_k"]).reset_index(drop=True)
    return df


def plot_coverage_vs_topk(df_main: pd.DataFrame, df_baseline: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("Coverage vs Top-K  |  NeuroSymbolic Union (FinDKG)", fontsize=14)

    splits = ["valid", "test"]
    colors = {"sym_coverage": "#E07B39", "neu_coverage": "#4A90D9", "union_coverage": "#2CA02C"}
    labels = {"sym_coverage": "Symbolic", "neu_coverage": "Neural", "union_coverage": "Union"}

    for ax, split in zip(axes, splits):
        df_split = df_main[df_main["split"] == split].sort_values("top_k")
        for col, color in colors.items():
            ax.plot(df_split["top_k"], df_split[col], marker="o", label=labels[col], color=color, linewidth=2)

        # Baseline dashed lines (vanilla uses split=valid for both panels as reference)
        b = df_baseline.iloc[0]
        ax.axhline(b["neu_coverage"], linestyle="--", color="#4A90D9", alpha=0.5, linewidth=1, label="Baseline Neural")
        ax.axhline(b["union_coverage"], linestyle="--", color="#2CA02C", alpha=0.5, linewidth=1, label="Baseline Union")

        ax.set_title(f"Split: {split}")
        ax.set_xlabel("Top-K")
        ax.set_ylabel("Coverage (hits / gold)")
        ax.set_xticks([10, 20, 30])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "coverage_vs_topk.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_coverage_bars(df_main: pd.DataFrame):
    df_test = df_main[df_main["split"] == "test"].sort_values("top_k").set_index("top_k")
    cols = ["sym_coverage", "neu_coverage", "union_coverage"]
    plot_df = df_test[cols].rename(columns={"sym_coverage": "Symbolic", "neu_coverage": "Neural", "union_coverage": "Union"})

    ax = plot_df.plot(kind="bar", figsize=(8, 5), color=["#E07B39", "#4A90D9", "#2CA02C"],
                      edgecolor="white", width=0.65)
    ax.set_title("Coverage by Method at Each Top-K  (Test Split)", fontsize=13)
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Coverage")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_xticklabels(["k=10", "k=20", "k=30"], rotation=0)
    ax.legend(title="Method")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in ax.patches:
        ax.annotate(f"{bar.get_height():.1%}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "coverage_bars_test.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_hit_counts(df_main: pd.DataFrame):
    df_test = df_main[df_main["split"] == "test"].sort_values("top_k").set_index("top_k")
    cols = ["sym_hit", "neu_hit", "union_hit"]
    plot_df = df_test[cols].rename(columns={"sym_hit": "Symbolic Hits", "neu_hit": "Neural Hits", "union_hit": "Union Hits"})

    ax = plot_df.plot(kind="bar", figsize=(8, 5), color=["#E07B39", "#4A90D9", "#2CA02C"],
                      edgecolor="white", width=0.65)
    ax.set_title("Hit Counts by Method at Each Top-K  (Test Split)", fontsize=13)
    ax.set_xlabel("Top-K")
    ax.set_ylabel("# Hits")
    ax.set_xticklabels(["k=10", "k=20", "k=30"], rotation=0)
    ax.legend(title="Method")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in ax.patches:
        ax.annotate(f"{int(bar.get_height())}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "hit_counts_test.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_metrics()
    print("\n=== Full Metrics DataFrame ===")
    print(df[["dir", "split", "top_k", "is_baseline", "sym_coverage", "neu_coverage", "union_coverage"]].to_string(index=False))

    df_baseline = df[df["is_baseline"]].copy()
    df_main = df[~df["is_baseline"]].copy()

    plot_coverage_vs_topk(df_main, df_baseline)
    plot_coverage_bars(df_main)
    plot_hit_counts(df_main)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
