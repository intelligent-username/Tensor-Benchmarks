"""Visualization of benchmark results.

Generates bar charts, box plots, and aggregate tables from bench_report.json.
Outputs are organized under results/bar, results/box, and results/table.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, MutableMapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, LogLocator, LogFormatterSciNotation, NullFormatter

REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "bench_report.json")


# Load results
def load_results(path: str = REPORT_PATH) -> List[Dict[str, Any]]:
    """Load and filter benchmark results from a JSON report file.

    Returns only entries that contain an op, an impl, and a numeric median.

    Args:
        path (str): Path to the JSON report file. Defaults to REPORT_PATH.

    Returns:
        List[Dict[str, Any]]: Filtered benchmark results.
    """
    with open(path, "r") as f:
        data = json.load(f)
    # Filter out error entries and incomplete blocks
    return [r for r in data if r.get("op") and r.get("median") is not None and r.get("impl")]


# Group results by op & shape
def group_results(results: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, Tuple[int, ...], Tuple[int, ...]], List[Dict[str, Any]]]:
    """Group benchmark results by operation and tensor shapes.

    Args:
        results (Sequence[Dict[str, Any]]): List of benchmark results.

    Returns:
        Dict[Tuple[str, Tuple[int, ...], Tuple[int, ...]], List[Dict[str, Any]]]: Grouped results.
    """
    grouped: DefaultDict[Tuple[str, Tuple[int, ...], Tuple[int, ...]], List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        key = (r["op"], tuple(r["shape_A"]), tuple(r["shape_B"] or []))
        grouped[key].append(r)
    return dict(grouped)


# --- helpers for clean log-scale axes and stable box plots ---
def _sanitize_log_values(values, floor=1e-12):
    """Sanitize log-scale values for plotting.

    Args:
        values (Iterable): Values to sanitize.
        floor (float): Minimum value to use. Defaults to 1e-12.

    Returns:
        List[float]: Sanitized values.
    """
    vals = [v for v in values if v is not None and np.isfinite(v) and v > 0]
    if not vals:
        return [floor]
    min_pos = max(min(vals), floor)
    return [v if (v is not None and np.isfinite(v) and v > 0) else min_pos * 0.5 for v in values]


def _apply_log_ticks(ax, values, axis="y", max_decades=6, pad=0.05):
    """Apply log-scale ticks to a plot axis.

    Args:
        ax: Matplotlib axis object.
        values: Values to determine tick positions.
        axis (str): Axis to apply ticks ('x' or 'y'). Defaults to 'y'.
        max_decades (int): Maximum number of decades to display. Defaults to 6.
        pad (float): Padding for axis limits. Defaults to 0.05.
    """
    vals = [v for v in values if v > 0]
    if not vals:
        return
    vmin = min(vals)
    vmax = max(vals)
    log_min = int(np.floor(np.log10(vmin)))
    log_max = int(np.ceil(np.log10(vmax)))
    span = max(1, log_max - log_min)
    step = max(1, int(np.ceil(span / max_decades)))
    ticks = [10 ** e for e in range(log_min, log_max + 1, step)]

    if axis == "y":
        ax.set_yscale("log")
        ax.set_ylim((10 ** log_min) * (1 - pad), (10 ** log_max) * (1 + pad))
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_xscale("log")
        ax.set_xlim((10 ** log_min) * (1 - pad), (10 ** log_max) * (1 + pad))
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_minor_formatter(NullFormatter())


# Plot comparison for each op/size
def plot_all(grouped: Dict[Tuple[str, Tuple[int, ...], Tuple[int, ...]], List[Dict[str, Any]]]) -> None:
    """Generate bar plots for all benchmark results.

    Args:
        grouped (Dict[Tuple[str, Tuple[int, ...], Tuple[int, ...]], List[Dict[str, Any]]]): Grouped benchmark results.
    """
    basedir = os.path.join(os.path.dirname(__file__), "..", "results")
    outdir = os.path.join(basedir, "bar")
    os.makedirs(outdir, exist_ok=True)
    op_names = {
        "ADD": "Matrix Addition",
        "MM": "Matrix Multiplication",
        "SCAL": "Scalar Multiplication",
        "DOT": "Dot Product",
        "BMM": "Batched Matrix Multiplication",
    }
    def shape_str(shape: Tuple[int, ...] | List[int] | None) -> str:
        return "x".join(str(s) for s in shape) if shape else "-"
    for key, group in grouped.items():
        op, shapeA, shapeB = key
        labels = []
        medians = []
        for impl in ["C", "NumPy", "PyTorch", "Vanilla"]:
            found = next((r for r in group if r["impl"] == impl), None)
            if found:
                labels.append(impl)
                medians.append(found["median"])
        if not medians:
            continue
        plt.figure(figsize=(7,5))
        ax = plt.gca()
        plt.bar(labels, medians, color=["#2c3e50", "#27ae60", "#2980b9", "#e67e22"])
        plt.ylabel("Median Time (s, log scale)")
        _apply_log_ticks(ax, [m for m in medians if m is not None], axis="y")
        op_title = op_names.get(op, op)
        title = f"{op_title}: {shape_str(shapeA)}"
        if shapeB and op not in ("SCAL",):
            title += f" vs {shape_str(shapeB)}"
        plt.title(title)
        for i, v in enumerate(medians):
            plt.text(i, v, f"{v:.2g}", ha="center", va="bottom")
        ax.grid(axis="y", which="both", linestyle="--", alpha=0.3)
        plt.tight_layout()
        fname = f"{op}_{shape_str(shapeA)}_{shape_str(shapeB)}.png"
        plt.savefig(os.path.join(outdir, fname))
        plt.close()


# Plot comparison for each op/size and generate aggregate table
def plot_all_and_table(grouped: Dict[Tuple[str, Tuple[int, ...], Tuple[int, ...]], List[Dict[str, Any]]]) -> None:
    """Generate bar plots and an aggregate table for benchmark results.

    Args:
        grouped (Dict[Tuple[str, Tuple[int, ...], Tuple[int, ...]], List[Dict[str, Any]]]): Grouped benchmark results.
    """
    basedir = os.path.join(os.path.dirname(__file__), "..", "results")
    bardir = os.path.join(basedir, "bar")
    tabledir = os.path.join(basedir, "table")
    os.makedirs(bardir, exist_ok=True)
    os.makedirs(tabledir, exist_ok=True)
    op_names = {
        "ADD": "Matrix Addition",
        "MM": "Matrix Multiplication",
        "SCAL": "Scalar Multiplication",
        "DOT": "Dot Product",
        "BMM": "Batched Matrix Multiplication",
    }
    def shape_str(shape: Tuple[int, ...] | List[int] | None) -> str:
        return "x".join(str(s) for s in shape) if shape else "-"
    # --- Bar plots ---
    for key, group in grouped.items():
        op, shapeA, shapeB = key
        labels = []
        medians = []
        for impl in ["C", "NumPy", "PyTorch", "Vanilla"]:
            found = next((r for r in group if r["impl"] == impl), None)
            if found:
                labels.append(impl)
                medians.append(found["median"])
        if not medians:
            continue
        plt.figure(figsize=(7,5))
        ax = plt.gca()
        plt.bar(labels, medians, color=["#2c3e50", "#27ae60", "#2980b9", "#e67e22"])
        plt.ylabel("Median Time (s, log scale)")
        _apply_log_ticks(ax, [m for m in medians if m is not None], axis="y")
        op_title = op_names.get(op, op)
        title = f"{op_title}: {shape_str(shapeA)}"
        if shapeB and op not in ("SCAL",):
            title += f" vs {shape_str(shapeB)}"
        plt.title(title)
        for i, v in enumerate(medians):
            plt.text(i, v, f"{v:.2g}", ha="center", va="bottom")
        ax.grid(axis="y", which="both", linestyle="--", alpha=0.3)
        plt.tight_layout()
        fname = f"{op}_{shape_str(shapeA)}_{shape_str(shapeB)}.png"
        plt.savefig(os.path.join(bardir, fname))
        plt.close()

    # --- Aggregate table (CSV and Markdown) ---
    table_rows = []
    header = ["Op", "Shape A", "Shape B", "C", "NumPy", "PyTorch", "Vanilla"]
    for key, group in grouped.items():
        op, shapeA, shapeB = key
        row = [op, shape_str(shapeA), shape_str(shapeB)]
        for impl in ["C", "NumPy", "PyTorch", "Vanilla"]:
            found = next((r for r in group if r["impl"] == impl), None)
            if found:
                row.append(f"{found['median']:.6g}")
            else:
                row.append("skipped")
        table_rows.append(row)
    csv_path = os.path.join(tabledir, "aggregate_table.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in table_rows:
            f.write(",".join(row) + "\n")
    md_path = os.path.join(tabledir, "aggregate_table.md")
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "---|"*len(header) + "\n")
        for row in table_rows:
            f.write("| " + " | ".join(row) + " |\n")


def plot_boxplots(grouped: Dict[Tuple[str, Tuple[int, ...], Tuple[int, ...]], List[Dict[str, Any]]]) -> None:
    """Generate box plots for benchmark results.

    Args:
        grouped (Dict[Tuple[str, Tuple[int, ...], Tuple[int, ...]], List[Dict[str, Any]]]): Grouped benchmark results.
    """
    basedir = os.path.join(os.path.dirname(__file__), "..", "results")
    outdir = os.path.join(basedir, "box")
    os.makedirs(outdir, exist_ok=True)
    impls = ["C", "NumPy", "PyTorch", "Vanilla"]
    impl_colors = {
        "C": "#2c3e50",
        "NumPy": "#27ae60",
        "PyTorch": "#2980b9",
        "Vanilla": "#e67e22",
    }
    op_names = {
        "ADD": "Matrix Addition",
        "MM": "Matrix Multiplication",
        "SCAL": "Scalar Multiplication",
        "DOT": "Dot Product",
        "BMM": "Batched Matrix Multiplication",
    }


    def shape_str(shape: Tuple[int, ...] | List[int] | None) -> str:
        return "x".join(str(s) for s in shape) if shape else "-"
    for key, group in grouped.items():
        op, shapeA, shapeB = key
        data = []
        labels = []
        box_colors = []
        for impl in impls:
            found = next((r for r in group if r["impl"] == impl and r.get("times")), None)
            if found and found["times"] and len(found["times"]) > 1:
                sanitized = _sanitize_log_values(found["times"])
                data.append(sanitized)
                labels.append(impl)
                box_colors.append(impl_colors[impl])
        if not data:
            continue
        plt.figure(figsize=(7,5))
        ax = plt.gca()
        box = plt.boxplot(data, patch_artist=True, tick_labels=labels, vert=True, showmeans=True)
        for patch, color in zip(box['boxes'], box_colors):
            patch.set_facecolor(color)
        plt.ylabel("Time (s, log scale)")
        all_vals = [v for arr in data for v in arr]
        _apply_log_ticks(ax, all_vals, axis="y")
        ax.grid(axis="y", which="both", linestyle="--", alpha=0.3)
        op_title = op_names.get(op, op)
        title = f"{op_title}: {shape_str(shapeA)}"
        if shapeB and op not in ("SCAL",):
            title += f" vs {shape_str(shapeB)}"
        plt.title(title + " (Box Plot)")
        plt.tight_layout()
        fname = f"{op}_{shape_str(shapeA)}_{shape_str(shapeB)}_boxplot.png"
        plt.savefig(os.path.join(outdir, fname))
        plt.close()


"""
Main function to load results, generate plots, and create tables.
"""
if __name__ == "__main__":
    results = load_results()
    grouped = group_results(results)
    print(f"Loaded {len(results)} results, {len(grouped)} unique op/size cases.")
    plot_all_and_table(grouped)
    plot_boxplots(grouped)
    print("Aggregate table written to results/table/aggregate_table.csv and results/table/aggregate_table.md")
    print("Bar graphs written to results/bar/")
    print("Box plots written to results/box/")
