import argparse
import json
import sys
import os
from pathlib import Path

# Add the root directory to Python path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.train.fasterrcnn import train_kfold


def load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser("Train Faster R-CNN on augmented K-Fold splits")
    ap.add_argument("--config", required=True, help="Path to train_fasterrcnn.json")
    ap.add_argument("--strategy", choices=["strict", "smoothed", "loose"], required=True)

    # Optional overrides
    ap.add_argument("--project_dir", default=None, help="Overrides cfg.project_dir")
    ap.add_argument("--out_dir", default=None, help="Overrides cfg.out_dir")
    ap.add_argument("--aug_root", default=None, help="Overrides cfg.aug_root")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    # Add new arguments for fine-tuning
    ap.add_argument("--init_from_prev", type=bool, default=None, help="Whether to initialize from previous weights")
    ap.add_argument("--prev_weights_template", default=None, help="Template path for previous weights")

    args = ap.parse_args()

    cfg = load_json(Path(args.config))
    cfg["strategy"] = args.strategy

    # Apply CLI overrides
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
    # Add new overrides
    if args.init_from_prev is not None:
        cfg["init_from_prev"] = args.init_from_prev
    if args.prev_weights_template is not None:
        cfg["prev_weights_template"] = args.prev_weights_template

    out = train_kfold(cfg)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()