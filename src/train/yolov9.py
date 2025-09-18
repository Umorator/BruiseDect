from pathlib import Path
import re
import json
import csv
import gc
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# ---------- helpers ----------

def _find_yaml_folds(yaml_dir: Path) -> List[Path]:
    """Return kfold_*.yaml files sorted by fold number."""
    yaml_files = list(yaml_dir.glob("kfold_*.yaml"))

    def fold_num(p: Path) -> int:
        m = re.search(r"kfold_(\d+)\.ya?ml$", p.name, re.I)
        return int(m.group(1)) if m else 10**9

    return sorted(yaml_files, key=fold_num)


class ModifiedDetect(nn.Module):
    """
    Wraps the original Ultralytics Detect module and inserts a Dropout
    before the last conv of EACH per-scale head branch (cv2[i]).
    Forwards all attributes and calls the original Detect.forward().
    """
    def __init__(self, original_detect: nn.Module, dropout_rate: float = 0.3):
        super().__init__()
        self.detect = original_detect  # keep original module

        # insert dropout into each branch, but avoid double-inserting
        patched = 0
        if hasattr(self.detect, "cv2"):
            for i, seq in enumerate(self.detect.cv2):
                layers = list(seq.children())
                if not layers:
                    continue
                # don't double insert if already present
                if len(layers) >= 2 and isinstance(layers[-2], nn.Dropout):
                    continue
                self.detect.cv2[i] = nn.Sequential(*layers[:-1], nn.Dropout(dropout_rate), layers[-1])
                patched += 1
        self._patched_branches = patched

    def forward(self, x):
        return self.detect(x)

    # forward attribute access like stride, anchors, etc.
    def __getattr__(self, name):
        if name == "detect":
            return super().__getattr__(name)
        return getattr(self.detect, name)


def _replace_last_detect_with_modified(model: YOLO, rate: float = 0.3) -> int:
    """
    Find the LAST Detect module in model.model/model and replace it
    with ModifiedDetect that contains per-branch dropout. Returns
    number of branches patched (usually 3 for P3/P4/P5).
    """
    root = getattr(model.model, "model", None) or model.model  # often an nn.Sequential
    last_idx, last_det = None, None
    # iterate by index to replace cleanly
    for idx, m in enumerate(root):
        if m.__class__.__name__.lower() == "detect":
            last_idx, last_det = idx, m
    if last_det is None:
        print("[yolov9] WARN: no Detect module found")
        return 0

    wrapped = ModifiedDetect(last_det, dropout_rate=rate)
    root[last_idx] = wrapped  # replace in-place
    return wrapped._patched_branches


def _metric_map50(results) -> float:
    """Robustly extract mAP@0.5 across Ultralytics versions."""
    if hasattr(results, "results_dict"):
        rd = results.results_dict
        for k in ("metrics/mAP50(B)", "metrics/mAP50", "map50"):
            if k in rd:
                try:
                    return float(rd[k])
                except Exception:
                    pass
    for name in ("map50", "mAP50"):
        if hasattr(results, name):
            try:
                return float(getattr(results, name))
            except Exception:
                pass
    # fallback: search any key containing "50"
    if hasattr(results, "results_dict"):
        for k, v in results.results_dict.items():
            if "50" in k:
                try:
                    return float(v)
                except Exception:
                    continue
    return float("nan")


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ---------- core ----------

def train_one_fold(
    yaml_path: Path,
    cfg: Dict[str, Any],
    project_dir: Path,
    run_name: str,
    pretrained_weights: Optional[Path] = None
) -> Tuple[float, str]:
    """
    Train a single fold and return (mAP50, run_dir).
    If pretrained_weights is provided, use it for fine-tuning.
    """
    weights    = cfg.get("weights", "yolov9m.pt")
    epochs     = int(cfg.get("epochs", 30))
    batch      = int(cfg.get("batch", 8))
    imgsz      = int(cfg.get("imgsz", 640))
    freeze     = int(cfg.get("freeze", 9))
    lr0        = float(cfg.get("lr0", 0.001))
    lrf        = float(cfg.get("lrf", 0.01))
    optimizer  = cfg.get("optimizer", "AdamW")
    workers    = int(cfg.get("workers", 8))
    cache      = bool(cfg.get("cache", True))
    dropout    = float(cfg.get("dropout", 0.3))
    device     = cfg.get("device", None)
    ultra_cfg  = cfg.get("ultra_cfg", None)
    use_dropout = bool(cfg.get("use_dropout", True))  # New: control dropout injection

    # Load model - use pretrained weights if provided
    model_weights = pretrained_weights.as_posix() if pretrained_weights else weights
    model = YOLO(model_weights)

    # Replace last Detect with ModifiedDetect if requested
    if use_dropout:
        patched = _replace_last_detect_with_modified(model, rate=dropout)
        print(f"[yolov9] ModifiedDetect: patched {patched} head branches")
    else:
        print(f"[yolov9] Skipping dropout injection for fine-tuning")

    # Ensure project dir exists
    _ensure_dir(project_dir)

    # Train
    results = model.train(
        data=yaml_path.as_posix(),
        batch=batch,
        freeze=freeze,
        lr0=lr0,
        lrf=lrf,
        optimizer=optimizer,
        epochs=epochs,
        workers=workers,
        cache=cache,
        imgsz=imgsz,
        device=device if device else None,
        project=project_dir.as_posix(),
        name=run_name,
        exist_ok=True,
        cfg=ultra_cfg if ultra_cfg else None,
    )

    map50 = _metric_map50(results)
    run_dir = (project_dir / run_name).as_posix()

    # Cleanup VRAM
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return map50, run_dir


def train_kfold(cfg: Dict[str, Any]) -> str:
    """
    Loop over kfold_*.yaml files in cfg['yaml_dir'], train each fold,
    and write summary CSV+JSON into cfg['out_dir'].
    Supports fine-tuning with pretrained weights per fold.
    """
    yaml_dir = Path(cfg["yaml_dir"])
    project  = Path(cfg.get("ultra_project_dir", r"P:/BruiseDet_Repo/outputs/training/yolov9"))
    out_dir  = Path(cfg.get("out_dir", project.as_posix()))
    pretrained_dir = Path(cfg.get("pretrained_dir", ""))  # New: directory with pretrained weights
    _ensure_dir(project)
    _ensure_dir(out_dir)

    yaml_files = _find_yaml_folds(yaml_dir)
    if not yaml_files:
        raise FileNotFoundError(f"No kfold_*.yaml found under: {yaml_dir}")

    fold_to_map: Dict[int, float | None] = {}
    fold_to_run: Dict[int, str] = {}

    for y in yaml_files:
        m = re.search(r"kfold_(\d+)\.ya?ml$", y.name, re.I)
        fold = int(m.group(1)) if m else len(fold_to_map) + 1
        run_name = f"k{fold}"
        
        # Check if pretrained weights exist for this fold
        pretrained_weights = None
        if pretrained_dir and pretrained_dir.exists():
            # Look for weights in pattern: best.pt, last.pt, or fold-specific
            possible_paths = [
                pretrained_dir / f"k{fold}" / "weights" / "best.pt",
                pretrained_dir / f"k{fold}" / "weights" / "last.pt",
                pretrained_dir / f"fold{fold}" / "weights" / "best.pt",
                pretrained_dir / f"fold{fold}" / "weights" / "last.pt",
            ]
            
            for weight_path in possible_paths:
                if weight_path.exists():
                    pretrained_weights = weight_path
                    break
        
        print(f"[yolov9] training fold {fold} -> {y.name}  (run: {run_name})")
        if pretrained_weights:
            print(f"[yolov9] using pretrained weights: {pretrained_weights}")
        
        m50, run_dir = train_one_fold(y, cfg, project, run_name, pretrained_weights)
        fold_to_map[fold] = float(m50) if m50 == m50 else None
        fold_to_run[fold] = run_dir

    vals = [v for v in fold_to_map.values() if v is not None]
    avg = float(np.mean(vals)) if vals else float("nan")
    std = float(np.std(vals)) if vals else float("nan")

    summary = {
        "strategy": cfg.get("strategy"),
        "yaml_dir": yaml_dir.as_posix(),
        "ultra_project_dir": project.as_posix(),
        "pretrained_dir": pretrained_dir.as_posix() if pretrained_dir else None,
        "weights": cfg.get("weights", "yolov9m.pt"),
        "epochs": int(cfg.get("epochs", 30)),
        "batch": int(cfg.get("batch", 8)),
        "imgsz": int(cfg.get("imgsz", 640)),
        "freeze": int(cfg.get("freeze", 9)),
        "lr0": float(cfg.get("lr0", 0.001)),
        "lrf": float(cfg.get("lrf", 0.01)),
        "optimizer": cfg.get("optimizer", "AdamW"),
        "workers": int(cfg.get("workers", 8)),
        "cache": bool(cfg.get("cache", True)),
        "dropout": float(cfg.get("dropout", 0.3)),
        "use_dropout": bool(cfg.get("use_dropout", True)),
        "device": cfg.get("device"),
        "ultra_cfg": cfg.get("ultra_cfg"),
        "folds": sorted(fold_to_map.keys()),
        "map50_per_fold": fold_to_map,
        "run_dir_per_fold": fold_to_run,
        "average_map50": avg,
        "std_map50": std,
    }

    # Write JSON + CSV
    (out_dir / "summary_yolov9.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "summary_yolov9.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fold", "mAP50", "run_dir"])
        for k in sorted(fold_to_map.keys()):
            w.writerow([k, fold_to_map[k], fold_to_run.get(k, "")])
        w.writerow([])
        w.writerow(["average_map50", avg])
        w.writerow(["std_map50", std])

    print(f"[yolov9] mAP@0.5 per fold: {fold_to_map}")
    print(f"[yolov9] average={avg:.4f}  std={std:.4f}")
    return out_dir.as_posix()