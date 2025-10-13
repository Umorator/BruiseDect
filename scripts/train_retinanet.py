import argparse
import json
import sys
import os
from pathlib import Path

# Add the root directory to Python path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.train.retinanet import train_kfold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to train_retinanet.json")
    p.add_argument("--strategy", choices=["strict", "smoothed", "loose"], required=True)
    # optional overrides
    p.add_argument("--project_dir", default=None, help="outputs root e.g. P:/.../RetinaNet (k1, k2 inside)")
    p.add_argument("--out_dir", default=None, help="alias of project_dir (kept for parity)")
    p.add_argument("--aug_root", default=None, help="Overrides cfg.aug_root")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None)
    # Hyperparameter overrides
    p.add_argument("--lr", type=float, default=None, help="Learning rate")
    p.add_argument("--lrf", type=float, default=None, help="Final learning rate factor")
    p.add_argument("--use_dropout", type=bool, default=None, help="Whether to use dropout in box predictor")
    p.add_argument("--dropout_prob", type=float, default=None, help="Dropout probability")
    p.add_argument("--weight_decay", type=float, default=None, help="Weight decay for optimizer")
    p.add_argument("--resize_number", type=int, default=None, help="Image resize dimension")
    # Add new arguments for fine-tuning
    p.add_argument("--init_from_prev", type=bool, default=None, help="Whether to initialize from previous weights")
    p.add_argument("--prev_weights_template", default=None, help="Template path for previous weights")

    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    cfg["strategy"] = args.strategy

    if args.project_dir is not None:
        cfg["project_dir"] = args.project_dir
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    if args.aug_root is not None:
        cfg["aug_root"] = args.aug_root
    if args.workers is not None:
        cfg["workers"] = args.workers
    if args.threads is not None:
        cfg["threads"] = args.threads
    if args.batch is not None:
        cfg["batch"] = args.batch
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.device is not None:
        cfg["device"] = args.device
    # Hyperparameter overrides
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.lrf is not None:
        cfg["lrf"] = args.lrf
    if args.use_dropout is not None:
        cfg["use_dropout"] = args.use_dropout
    if args.dropout_prob is not None:
        cfg["dropout_prob"] = args.dropout_prob
    if args.weight_decay is not None:
        cfg["weight_decay"] = args.weight_decay
    if args.resize_number is not None:
        cfg["resize_number"] = args.resize_number
    # Add new overrides
    if args.init_from_prev is not None:
        cfg["init_from_prev"] = args.init_from_prev
    if args.prev_weights_template is not None:
        cfg["prev_weights_template"] = args.prev_weights_template

    train_kfold(cfg)


if __name__ == "__main__":
    main()