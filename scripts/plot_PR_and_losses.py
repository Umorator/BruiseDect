#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys
import numpy as np

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from eval.pr_curves import (
    _load_cfg, build_coco_from_dirs, coco_pr_curve, 
    plot_pr_curves_with_std, plot_pr_curves_all_strategies_with_std
)
from eval.train_val_losses import (
    load_config, plot_loss_curves_with_std, process_training_data
)


def main():
    ap = argparse.ArgumentParser(description="Generate PR curves and loss curves")
    ap.add_argument("--config", required=True, help="configs/evaluate_prediction.json")
    ap.add_argument("--strategy", nargs="*", help="Override strategies e.g. Strict Smoothed Loose")
    ap.add_argument("--model", nargs="*", help="Override models e.g. YOLO FRCNN RetinaNet")
    ap.add_argument("--folds", nargs="*", type=int, help="Only these folds e.g. 1 2 3")
    ap.add_argument("--iou", type=float, default=None, help="IoU threshold for PR (default uses cfg)")
    ap.add_argument("--base-path", default="outputs/training", 
                   help="Base path containing training outputs (default: outputs/training)")
    ap.add_argument("--plot-type", choices=["pr", "loss", "both"], default="both",
                   help="Type of plots to generate: pr, loss, or both")
    args = ap.parse_args()

    cfg = _load_cfg(Path(args.config))
    strategies = [s.capitalize() for s in (args.strategy or cfg["strategies"])]
    models = args.model or cfg["models"]
    folds = args.folds or list(range(1, 6))
    iou_thr = args.iou if args.iou is not None else float(cfg.get("iou_threshold", 0.5))
    base_path = Path(args.base_path)

    # Create output directories under outputs/eval/plots/
    eval_root = Path(cfg.get("out_dir", "outputs/eval")) / "plots"
    pr_out_root = eval_root / "pr_curves"
    loss_out_root = eval_root / "loss_curves"
    pr_out_root.mkdir(parents=True, exist_ok=True)
    loss_out_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] Strategies: {strategies}")
    print(f"[info] Models: {models}, Folds: {folds}")
    print(f"[info] Plot type: {args.plot_type}")

    # Generate PR curves
    if args.plot_type in ["pr", "both"]:
        print("\n[info] Generating PR curves...")
        aug_tmpl = cfg["aug_root_template"]      # data/augmented/{strategy}
        pred_tmpl = cfg["pred_root_template"]     # outputs/predictions/{strategy}/{model}
        model_norm = cfg.get("model_norm", {"YOLO": "orig", "FRCNN": "resized", "RetinaNet": "resized"})
        target_size = int(cfg.get("resize_number", 640))

        # One plot per strategy (overlay models), plus an ALL plot overlaying everything
        all_curves_data = []
        for strat in strategies:
            strat_curves_data = []
            for model in models:
                policy = model_norm.get(model, "orig")
                all_Ps = []  # Store all P arrays for this model-strategy combination
                R_ref = None

                for k in folds:
                    gt_labels = Path(aug_tmpl.format(strategy=strat)) / f"Kfold_{k}" / "val" / "labels"
                    pred_dir = Path(pred_tmpl.format(strategy=strat, model=model)) / f"k{k}"

                    if not gt_labels.is_dir() or not pred_dir.is_dir():
                        print(f"[warn] missing data — GT:{gt_labels} | PRED:{pred_dir}; skipping fold {k}")
                        continue

                    try:
                        coco_gt, coco_pred = build_coco_from_dirs(
                            gt_labels_dir=gt_labels,
                            pred_dir=pred_dir,
                            norm_policy=policy,
                            target_size=target_size
                        )
                        R, P = coco_pr_curve(coco_gt, coco_pred, iou_thr=iou_thr)
                        R_ref = R if R_ref is None else R_ref
                        
                        # Ensure all P arrays have the same length
                        if len(P) == len(R_ref):
                            all_Ps.append(P)
                        else:
                            print(f"[warn] Mismatched array lengths for {model} {strat} fold {k}: {len(P)} vs {len(R_ref)}")
                            
                    except Exception as e:
                        print(f"[error] Failed to process {model} {strat} fold {k}: {e}")
                        continue

                if len(all_Ps) < 2:  # Need at least 2 folds to compute std
                    print(f"[warn] Insufficient data for {model} ({strat}): {len(all_Ps)} folds")
                    continue

                # Stack all P arrays and compute mean and std
                P_stack = np.vstack([np.array(p, dtype=float) for p in all_Ps])
                P_mean = np.nanmean(P_stack, axis=0)
                P_std = np.nanstd(P_stack, axis=0)
                
                label = f"{model} ({strat})"
                strat_curves_data.append((label, R_ref, P_mean, P_std))
                all_curves_data.append((label, R_ref, P_mean, P_std))
                
                print(f"[info] {label}: {len(all_Ps)} folds, mean AP: {np.mean(P_mean):.3f} ± {np.mean(P_std):.3f}")

            if strat_curves_data:
                out_file = pr_out_root / f"PR_{strat}_IoU{str(iou_thr).replace('.','')}_with_std.png"
                plot_pr_curves_with_std(strat_curves_data, out_file, 
                                       title=f"Precision–Recall @ IoU={iou_thr} ({strat})")
                print(f"[pr] wrote {out_file}")

        if all_curves_data:
            out_file = pr_out_root / f"PR_ALL_IoU{str(iou_thr).replace('.','')}_with_std.png"
            plot_pr_curves_all_strategies_with_std(all_curves_data, out_file, 
                                                 title=f"Precision–Recall @ IoU={iou_thr}")
            print(f"[pr] wrote {out_file}")

    # Generate loss curves
    if args.plot_type in ["loss", "both"]:
        print("\n[info] Generating loss curves...")
        
        # Process data and create plots
        all_curves_data = process_training_data(base_path, strategies, models, folds)

        # Plot individual strategies
        for strategy, strategy_data in all_curves_data.items():
            if strategy_data:  # Only plot if there's data
                out_file = loss_out_root / f"loss_curves_{strategy}_with_std.png"
                plot_loss_curves_with_std(
                    {strategy: strategy_data}, 
                    out_file,
                    title=f"Training and Validation Losses ({strategy})"
                )
                print(f"[loss] wrote {out_file}")

        # Plot all strategies together
        if len(all_curves_data) > 1:
            out_file = loss_out_root / "loss_curves_ALL_with_std.png"
            plot_loss_curves_with_std(
                all_curves_data,
                out_file,
                title="Training and Validation Losses (All Strategies)"
            )
            print(f"[loss] wrote {out_file}")

    print("\n[info] Evaluation complete!")


if __name__ == "__main__":
    main()