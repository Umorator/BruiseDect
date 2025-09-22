#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from eval.train_val_losses import (
    load_config, plot_loss_curves_with_std, process_training_data
)

def main():
    ap = argparse.ArgumentParser(description="Plot training and validation loss curves with standard deviation")
    ap.add_argument("--config", required=True, help="Path to config file (e.g., configs/evaluate_predictions.json)")
    ap.add_argument("--strategy", nargs="*", help="Strategies to plot (e.g., Strict Smoothed Loose)")
    ap.add_argument("--model", nargs="*", help="Models to plot (e.g., YOLO FRCNN RetinaNet)")
    ap.add_argument("--folds", nargs="*", type=int, help="Folds to include (e.g., 1 2 3 4 5)")
    ap.add_argument("--base-path", default="outputs/training", 
                   help="Base path containing training outputs (default: outputs/training)")
    args = ap.parse_args()

    # Load configuration
    cfg = load_config(Path(args.config))
    strategies = [s.capitalize() for s in (args.strategy or cfg["strategies"])]
    models = args.model or cfg["models"]
    folds = args.folds or list(range(1, 6))
    base_path = Path(args.base_path)

    # Output directory
    out_root = Path(cfg.get("out_dir", "outputs/eval")) / "loss_curves"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] Plotting loss curves for strategies: {strategies}")
    print(f"[info] Models: {models}, Folds: {folds}")

    # Process data and create plots
    all_curves_data = process_training_data(base_path, strategies, models, folds)

    # Plot individual strategies
    for strategy, strategy_data in all_curves_data.items():
        if strategy_data:  # Only plot if there's data
            out_file = out_root / f"loss_curves_{strategy}_with_std.png"
            plot_loss_curves_with_std(
                {strategy: strategy_data}, 
                out_file,
                title=f"Training and Validation Losses ({strategy})"
            )
            print(f"[loss] wrote {out_file}")

    # Plot all strategies together
    if len(all_curves_data) > 1:
        out_file = out_root / "loss_curves_ALL_with_std.png"
        plot_loss_curves_with_std(
            all_curves_data,
            out_file,
            title="Training and Validation Losses (All Strategies)"
        )
        print(f"[loss] wrote {out_file}")

if __name__ == "__main__":
    main()