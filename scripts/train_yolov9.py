import argparse
import json
import os
from pathlib import Path
import sys

# Add the root directory to Python path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.train.yolov9 import train_kfold

# Pragmatic Windows OpenMP duplicate workaround
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _cap_strategy(s: str | None) -> str | None:
    if not s:
        return s
    return s.capitalize()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train YOLOv9 across K-Fold YAMLs")
    ap.add_argument("--config", required=True, help="Path to train_yolov9.json")

    # Optional overrides
    ap.add_argument("--strategy", choices=["strict", "smoothed", "loose"], default=None)
    ap.add_argument("--yaml_dir", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--freeze", type=int, default=None)
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--lrf", type=float, default=None)
    ap.add_argument("--optimizer", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cache", type=str, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--use_dropout", type=str, default=None)  # New: control dropout
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", default=None)
    
    # New: for fine-tuning
    ap.add_argument("--pretrained_dir", default=None, 
                   help="Directory containing pretrained weights for each fold")
    ap.add_argument("--training_type", choices=["train", "fine_tune"], default="train",
                   help="Type of training: 'train' or 'fine_tune'")

    # Ultralytics-specific
    ap.add_argument("--ultra_project_dir", default=None)
    ap.add_argument("--ultra_cfg", default=None)

    args = ap.parse_args(argv)
    cfg = _load_json(Path(args.config))

    # Helper to write overrides into cfg if provided on CLI
    def put(k, v):
        if v is not None:
            cfg[k] = v

    put("strategy", args.strategy or cfg.get("strategy"))
    put("yaml_dir", args.yaml_dir)
    put("weights", args.weights)
    put("epochs", args.epochs)
    put("batch", args.batch)
    put("imgsz", args.imgsz)
    put("freeze", args.freeze)
    put("lr0", args.lr0)
    put("lrf", args.lrf)
    put("optimizer", args.optimizer)
    put("workers", args.workers)
    if args.cache is not None:
        cfg["cache"] = args.cache.lower() == "true"
    put("dropout", args.dropout)
    if args.use_dropout is not None:
        cfg["use_dropout"] = args.use_dropout.lower() == "true"
    put("device", args.device)
    put("out_dir", args.out_dir)
    put("pretrained_dir", args.pretrained_dir)  # New
    put("training_type", args.training_type)    # New
    put("ultra_project_dir", args.ultra_project_dir)
    put("ultra_cfg", args.ultra_cfg)

    # Resolve {strategy} templates
    strategy = cfg.get("strategy")
    cap_strategy = _cap_strategy(strategy)
    training_type = cfg.get("training_type", "train")

    for key in ("yaml_dir_template", "out_dir", "ultra_project_dir", "pretrained_dir"):
        val = cfg.get(key)
        if isinstance(val, str) and "{strategy}" in val:
            if not cap_strategy:
                raise ValueError(f"{key} uses '{{strategy}}' but no --strategy provided.")
            cfg[key] = val.replace("{strategy}", cap_strategy)

    # Defaults if not provided
    if not cfg.get("yaml_dir"):
        tpl = cfg.get("yaml_dir_template")
        if not tpl:
            raise ValueError("Provide 'yaml_dir' or 'yaml_dir_template' in config or CLI.")
        cfg["yaml_dir"] = tpl

    # Set appropriate project directory based on training type
    if not cfg.get("ultra_project_dir"):
        if training_type == "fine_tune":
            # Fine-tuning output directory
            cfg["ultra_project_dir"] = fr"outputs/training_ft/{cap_strategy}/yolov9"
        else:
            # Regular training output directory
            cfg["ultra_project_dir"] = fr"outputs/training/{cap_strategy}/yolov9"

    # out_dir: summary files; default alongside the runs
    if not cfg.get("out_dir"):
        cfg["out_dir"] = cfg["ultra_project_dir"]

    # Set dropout usage based on training type if not explicitly set
    if "use_dropout" not in cfg:
        cfg["use_dropout"] = (training_type == "train")  # Use dropout for training, not for fine-tuning

    out = train_kfold(cfg)
    print(f"[yolov9] summary written to: {out}")


if __name__ == "__main__":
    main()