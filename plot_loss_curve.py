#!/usr/bin/env python3
import argparse
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log_file(log_file_path):
    steps = []
    losses = []

    path = Path(log_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {log_file_path}")

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()

            if "loss" not in line or "global_step/max_steps" not in line:
                continue

            try:
                data = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue

            if "loss" not in data or "global_step/max_steps" not in data:
                continue

            try:
                loss = float(data["loss"])
                step_str = str(data["global_step/max_steps"])
                step = int(step_str.split("/")[0])
            except (ValueError, TypeError, IndexError):
                continue

            steps.append(step)
            losses.append(loss)

    if not steps:
        raise ValueError("No valid loss/step pairs were found in the log file.")

    return steps, losses


def plot_loss_curve(steps, losses, output_image_path):
    paired = sorted(zip(steps, losses), key=lambda x: x[0])
    steps_sorted = [x[0] for x in paired]
    losses_sorted = [x[1] for x in paired]

    plt.figure(figsize=(10, 6))
    plt.plot(steps_sorted, losses_sorted, linewidth=2)
    plt.title("KD Training Loss vs Step")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot loss vs step from a training log.")
    parser.add_argument("log_file", help="Path to the log file")
    parser.add_argument("-o", "--output", default="loss_curve.png", help="Output image path")
    args = parser.parse_args()

    steps, losses = parse_log_file(args.log_file)
    print(f"Found {len(steps)} points.")
    plot_loss_curve(steps, losses, args.output)
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()