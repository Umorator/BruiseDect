import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from src.predict.yolov9 import run_kfold_predictions


def _load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _override(cfg: dict, key: str, value):
    if value is not None:
        cfg[key] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to predict_yolov9.json")
    ap.add_argument("--strategy", choices=["strict", "smoothed", "loose"], default=None)
    ap.add_argument("--pred_out_dir", default=None, help="Override output predictions root")
    ap.add_argument("--train_project_dir", default=None, help="Override training project root (where k*/weights/best.pt live)")
    ap.add_argument("--device", default=None, help="cpu or cuda:0 etc.")
    args = ap.parse_args()

    cfg = _load_json(Path(args.config))
    _override(cfg, "strategy", args.strategy)
    _override(cfg, "pred_out_template", args.pred_out_dir)
    _override(cfg, "train_project_template", args.train_project_dir)
    _override(cfg, "device", args.device)

    # Normalize strategy capitalization for templates, but pass lower in cfg
    strategy = str(cfg.get("strategy", "strict")).lower()
    cfg["strategy"] = strategy

    # Run
    run_kfold_predictions(cfg)


if __name__ == "__main__":
    main()
