"""
Minimal CSV + stdout logger. No external dependencies.

  Logger(log_dir, run_name)
    .log(step, **metrics)       -- print to stdout, append to log_dir/run_name.csv
    .log_config(config_dict)    -- save as log_dir/run_name_config.yaml
    .plot(metric_keys, save_path=None)
      -- matplotlib line plot of logged metrics; saves or displays
"""

import csv
import os
from collections import defaultdict
from typing import Any


class Logger:
    def __init__(self, log_dir: str, run_name: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir  = log_dir
        self.run_name = run_name
        self.csv_path = os.path.join(log_dir, f"{run_name}.csv")
        self._header_written = False
        self._data: dict[str, list] = defaultdict(list)   # key -> [(step, val), ...]

    def log(self, step: int, **metrics: float) -> None:
        """Print to stdout and append a row to the CSV."""
        # Omit epoch_time_s from stdout to keep output readable; it is always written to CSV.
        display = {k: v for k, v in metrics.items() if k != "epoch_time_s"}
        parts = [f"step={step:>4}"] + [f"{k}={v:.4f}" for k, v in display.items()]
        print("  ".join(parts))

        if not self._header_written:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step"] + list(metrics.keys()))
            self._header_written = True

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step] + list(metrics.values()))

        for k, v in metrics.items():
            self._data[k].append((step, v))

    def log_config(self, config_dict: dict) -> None:
        """Persist the experiment config next to the CSV."""
        import yaml
        path = os.path.join(self.log_dir, f"{self.run_name}_config.yaml")
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def plot(self, metric_keys: list[str], save_path: str | None = None) -> None:
        """Line plot of requested metrics. Saves to save_path or shows interactively."""
        import matplotlib.pyplot as plt

        n = len(metric_keys)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, key in zip(axes, metric_keys):
            if key in self._data:
                steps, vals = zip(*self._data[key])
                ax.plot(steps, vals, marker="o", markersize=3)
            ax.set_title(key)
            ax.set_xlabel("step")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()
