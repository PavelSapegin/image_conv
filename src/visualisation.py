import json
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_benchmarks(filepath: str) -> pd.DataFrame:
    """Parse benchmark results from JSON file and return a DataFrame."""
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    records = []

    size_pattern = re.compile(r"(\d+x\d+)")

    for bench in data.get("benchmarks", []):
        name = bench.get("name", "")

        if "custom" in name:
            impl = "custom"
        elif "pillow" in name:
            impl = "pillow"
        elif "opencv" in name:
            impl = "opencv"
        else:
            impl = "unknown"

        p = bench.get("params", {})

        raw_size = p.get("image_data")
        if raw_size:
            image_size = f"{raw_size}x{raw_size}"
        else:
            size_match = size_pattern.search(name)
            image_size = size_match.group(1) if size_match else "unknown"

        kernel = p.get("kernel_name", "unknown")
        padding = p.get("padding_mode", "unknown")

        is_gray = str(p.get("gray", "")).lower() == "true"
        color_mode = "gray" if is_gray else "color"

        stats = bench.get("stats", {})
        all_runs = stats.get("data", [])

        if not all_runs:
            mean_time = stats.get("mean")
            if mean_time is not None:
                all_runs = [mean_time]
            else:
                continue

        for run_time in all_runs:
            records.append(
                {
                    "implementation": impl,
                    "image_size": image_size,
                    "kernel": kernel,
                    "padding": padding,
                    "color_mode": color_mode,
                    "time_ms": run_time * 1000,
                }
            )

    return pd.DataFrame(records)


def plot_results(df: pd.DataFrame, output_filename: str = "bench_results.png") -> None:
    """Draw bar plots comparing implementations across different parameters."""

    df = df[df["image_size"] != "unknown"].copy()

    if df.empty:
        print("Error: No valid benchmark data to plot.")
        sys.exit(1)

    expected_sizes = ["128x128", "512x512", "1024x1024", "4096x4096"]
    actual_sizes = [s for s in expected_sizes if s in df["image_size"].unique()]
    df["image_size"] = pd.Categorical(
        df["image_size"], categories=actual_sizes, ordered=True
    )

    sns.set_theme(style="whitegrid")

    g = sns.catplot(
        data=df,
        x="image_size",
        y="time_ms",
        hue="implementation",
        col="color_mode",
        row="kernel",
        kind="bar",
        errorbar="sd",
        capsize=0.1,
        err_kws={"linewidth": 1.5},
        edgecolor=".2",
        height=4,
        aspect=1.2,
        sharey=False,
    )

    for ax in g.axes.flat:
        ax.set_yscale("log")
        ax.yaxis.grid(True, which="both", linestyle="--", alpha=0.5)

    g.set_axis_labels("Image Size", "Execution Time (ms)")
    g.figure.suptitle("Comparison of Implementation Performance", y=1.02, fontsize=16)

    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    print(f"Graph successfully saved to {output_filename}")
    plt.show()


if __name__ == "__main__":
    df_results = parse_benchmarks("result.json")
    plot_results(df_results)
