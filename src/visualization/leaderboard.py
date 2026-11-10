"""
Leaderboard builder and renderer.

Aggregates evaluation results from all models and tasks into:
  - Console table (via rich)
  - LaTeX table (for paper submission)
  - Radar chart (multi-metric comparison)
  - Per-task bar charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from pathlib import Path
from typing import Dict, List, Optional
from rich.table import Table
from rich.console import Console
from rich import box


METRIC_DISPLAY = {
    "gaussian_nll": "Gaussian NLL ↓",
    "rmse": "RMSE ↓",
    "crps": "CRPS ↓",
    "ece": "ECE ↓",
    "coverage_90": "Coverage 90% →1",
    "coverage_50": "Coverage 50% →1",
    "sharpness": "Sharpness ↓",
    "interval_score_90": "IS-90 ↓",
}

MODEL_COLORS = {
    "pinn": "#185FA5",
    "mc_dropout": "#1D9E75",
    "deep_gp": "#7F77DD",
    "refann": "#D85A30",
    "mcmc_bnn": "#BA7517",
}


class Leaderboard:
    """
    Maintains a table of (model, task) → UQMetrics and renders views.

    Usage:
        lb = Leaderboard()
        lb.add_result("pinn", "heat_equation", metrics_dict)
        lb.add_result("mc_dropout", "heat_equation", metrics_dict)
        lb.show()
        lb.save("results/leaderboard.csv")
    """

    def __init__(self):
        self.records: List[dict] = []

    def add_result(self, model_name: str, task_name: str, metrics: dict):
        self.records.append({"model": model_name, "task": task_name, **metrics})

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def show(self, task: Optional[str] = None, metrics: Optional[List[str]] = None):
        """Print a rich table to the console."""
        df = self.to_dataframe()
        if task:
            df = df[df["task"] == task]
        if metrics is None:
            metrics = ["gaussian_nll", "rmse", "crps", "ece", "coverage_90"]

        console = Console()
        table = Table(
            title=f"UQ Benchmark Leaderboard — {task or 'all tasks'}",
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold",
        )
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Task", style="dim")
        for m in metrics:
            table.add_column(METRIC_DISPLAY.get(m, m), justify="right")

        for _, row in df.sort_values("gaussian_nll").iterrows():
            table.add_row(
                row["model"],
                row["task"],
                *[f"{row.get(m, float('nan')):.4f}" for m in metrics],
            )

        console.print(table)

    def save(self, path: str):
        df = self.to_dataframe()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Leaderboard saved to {path}")

    def to_latex(self, task: Optional[str] = None) -> str:
        df = self.to_dataframe()
        if task:
            df = df[df["task"] == task]
        metrics = ["gaussian_nll", "rmse", "crps", "ece", "coverage_90"]
        df = df.sort_values("gaussian_nll")

        lines = ["\\begin{tabular}{l" + "r" * len(metrics) + "}"]
        lines.append("\\toprule")
        header = "Model & " + " & ".join([METRIC_DISPLAY.get(m, m) for m in metrics]) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")

        for _, row in df.iterrows():
            vals = [f"{row.get(m, float('nan')):.4f}" for m in metrics]
            lines.append(row["model"] + " & " + " & ".join(vals) + " \\\\")

        lines += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(lines)

    def plot_radar(self, task: str, save_path: Optional[str] = None) -> plt.Figure:
        """Multi-metric radar chart comparing all models on a single task."""
        df = self.to_dataframe()
        df = df[df["task"] == task].set_index("model")

        radar_metrics = ["gaussian_nll", "rmse", "crps", "ece", "interval_score_90"]
        df_radar = df[radar_metrics].copy()

        # Normalise each metric to [0,1] (0 = best)
        for col in radar_metrics:
            vmin, vmax = df_radar[col].min(), df_radar[col].max()
            if vmax > vmin:
                df_radar[col] = (df_radar[col] - vmin) / (vmax - vmin)
            else:
                df_radar[col] = 0.0

        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
        for model_name, row in df_radar.iterrows():
            vals = row.tolist() + [row.iloc[0]]
            color = MODEL_COLORS.get(model_name, "#888")
            ax.plot(angles, vals, "o-", lw=2, label=model_name, color=color)
            ax.fill(angles, vals, alpha=0.07, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace("_", "\n") for m in radar_metrics], fontsize=9)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=7)
        ax.set_title(f"Radar: {task}", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_bar_comparison(
        self, metric: str = "gaussian_nll", save_path: Optional[str] = None
    ) -> plt.Figure:
        """Side-by-side bars per task, grouped by model."""
        df = self.to_dataframe()
        tasks = df["task"].unique()
        models = df["model"].unique()

        x = np.arange(len(tasks))
        width = 0.8 / len(models)
        fig, ax = plt.subplots(figsize=(max(8, 3 * len(tasks)), 5))

        for i, model in enumerate(models):
            vals = [
                df[(df["model"] == model) & (df["task"] == t)][metric].values
                for t in tasks
            ]
            heights = [v[0] if len(v) > 0 else np.nan for v in vals]
            offset = (i - len(models) / 2 + 0.5) * width
            color = MODEL_COLORS.get(model, "#888")
            ax.bar(x + offset, heights, width=width * 0.9, label=model, color=color, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(tasks, fontsize=10)
        ax.set_ylabel(METRIC_DISPLAY.get(metric, metric))
        ax.set_title(f"Benchmark: {metric} by task and model")
        ax.legend(fontsize=9)
        fig.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    @classmethod
    def load(cls, path: str) -> "Leaderboard":
        lb = cls()
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            lb.records.append(row.to_dict())
        return lb
