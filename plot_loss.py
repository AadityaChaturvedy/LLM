#!/usr/bin/env python3
import argparse
import re
from typing import List, Tuple

import matplotlib.pyplot as plt


LOSS_PATTERN = re.compile(
    r"(?:^|\b)Step\s+(\d+).*?\bLoss:\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
LOSS_ONLY_PATTERN = re.compile(r"\bLoss:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def parse_log(path: str) -> Tuple[List[int], List[float]]:
    steps: List[int] = []
    losses: List[float] = []
    inferred_step = 0

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = LOSS_PATTERN.search(line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                steps.append(step)
                losses.append(loss)
                inferred_step = max(inferred_step, step)
                continue

            match = LOSS_ONLY_PATTERN.search(line)
            if match:
                inferred_step += 1
                steps.append(inferred_step)
                losses.append(float(match.group(1)))

    return steps, losses


def smooth_series(steps: List[int], values: List[float], window: int) -> Tuple[List[int], List[float]]:
    if window <= 1 or len(values) < window:
        return steps, values

    smoothed: List[float] = []
    window_sum = sum(values[:window])
    smoothed.append(window_sum / window)
    for idx in range(window, len(values)):
        window_sum += values[idx] - values[idx - window]
        smoothed.append(window_sum / window)

    return steps[window - 1 :], smoothed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot loss curves from two training log files."
    )
    parser.add_argument("log_a", help="Path to first log file")
    parser.add_argument("log_b", nargs="?", default=None, help="Path to second log file (optional)")
    parser.add_argument("--label-a", default="Run A", help="Legend label for first log")
    parser.add_argument("--label-b", default="Run B", help="Legend label for second log")
    parser.add_argument(
        "--output",
        default="loss_comparison.png",
        help="Output image path (png, svg, etc.)",
    )
    parser.add_argument("--title", default="Loss Comparison", help="Plot title")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=0,
        help="Moving average window (0 or 1 disables smoothing)",
    )
    parser.add_argument(
        "--delta-output",
        default="loss_delta.png",
        help="Output image path for loss delta (Run B - Run A)",
    )
    parser.add_argument("--dpi", type=int, default=140, help="Output DPI")
    args = parser.parse_args()

    steps_a, losses_a = parse_log(args.log_a)
    steps_b, losses_b = parse_log(args.log_b) if args.log_b else ([], [])

    if not steps_a:
        raise SystemExit("No loss values found in the first log file.")

    plt.figure(figsize=(10, 6))
    plt.plot(steps_a, losses_a, label=f"{args.label_a} (raw)", linewidth=1.4, alpha=0.6)
    if steps_b:
        plt.plot(steps_b, losses_b, label=f"{args.label_b} (raw)", linewidth=1.4, alpha=0.6)

    if args.smooth_window > 1:
        smooth_steps_a, smooth_losses_a = smooth_series(steps_a, losses_a, args.smooth_window)
        plt.plot(
            smooth_steps_a,
            smooth_losses_a,
            label=f"{args.label_a} (smoothed)",
            linewidth=2.2,
        )
        if steps_b:
            smooth_steps_b, smooth_losses_b = smooth_series(steps_b, losses_b, args.smooth_window)
            plt.plot(
                smooth_steps_b,
                smooth_losses_b,
                label=f"{args.label_b} (smoothed)",
                linewidth=2.2,
            )
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(args.title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi)

    steps_a_map = dict(zip(steps_a, losses_a))
    steps_b_map = dict(zip(steps_b, losses_b))
    common_steps = sorted(set(steps_a_map) & set(steps_b_map))
    if common_steps:
        delta_values = [steps_b_map[step] - steps_a_map[step] for step in common_steps]
        plt.figure(figsize=(10, 4.8))
        plt.plot(common_steps, delta_values, label="Run B - Run A", linewidth=1.4)
        if args.smooth_window > 1:
            smooth_steps_d, smooth_delta = smooth_series(common_steps, delta_values, args.smooth_window)
            plt.plot(
                smooth_steps_d,
                smooth_delta,
                label="Smoothed delta",
                linewidth=2.2,
            )
        plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        plt.xlabel("Step")
        plt.ylabel("Loss delta")
        plt.title("Loss Delta (Run B - Run A)")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.delta_output, dpi=args.dpi)


if __name__ == "__main__":
    main()
