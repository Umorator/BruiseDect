import os
from pathlib import Path
from typing import List

import torch
from ultralytics import YOLO


def ioa(boxA, boxB):
    """
    IOA between two boxes in normalized [x,y,w,h] (center format).
    Uses intersection / min(areaA, areaB).
    """
    xA = max(boxA[0] - boxA[2] / 2, boxB[0] - boxB[2] / 2)
    yA = max(boxA[1] - boxA[3] / 2, boxB[1] - boxB[3] / 2)
    xB = min(boxA[0] + boxA[2] / 2, boxB[0] + boxA[2] / 2)  # <-- bug fix below
    yB = min(boxA[1] + boxA[3] / 2, boxB[1] + boxB[3] / 2)

    # correct xB calc (typo fix)
    xB = min(boxA[0] + boxA[2] / 2, boxB[0] + boxB[2] / 2)

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    smallest = min(areaA, areaB)
    return inter_area / smallest if smallest > 0 else 0.0


def _save_predictions_for_batch(results, image_paths: List[str], save_folder: Path, ioa_threshold: float = 0.5):
    save_folder.mkdir(parents=True, exist_ok=True)

    for res, img_path in zip(results, image_paths):
        base = Path(img_path).stem
        out_txt = save_folder / f"{base}.txt"

        h, w = res.orig_shape
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            # save empty file for completeness
            out_txt.write_text("")
            continue

        xywh_pix = boxes.xywh.tolist()     # pixel cx,cy,w,h
        confs = boxes.conf.tolist()
        clses = boxes.cls.tolist()

        # normalize to [0..1]
        xywh_norm = []
        for cx, cy, bw, bh in xywh_pix:
            xywh_norm.append([cx / w, cy / h, bw / w, bh / h])

        # IOA filtering (keep a subset)
        kept_idx = []
        for i, b in enumerate(xywh_norm):
            drop = False
            for j in kept_idx:
                if ioa(b, xywh_norm[j]) > ioa_threshold:
                    drop = True
                    break
            if not drop:
                kept_idx.append(i)

        with out_txt.open("w") as f:
            for i in kept_idx:
                c = int(clses[i])
                x, y, bw, bh = xywh_norm[i]
                conf = float(confs[i])
                f.write(f"{c} {x} {y} {bw} {bh} {conf}\n")


def _collect_folds(aug_root: Path, strategy_cap: str) -> List[Path]:
    root = aug_root / strategy_cap
    if not root.exists():
        raise FileNotFoundError(f"No augmented data for strategy '{strategy_cap}' at {root}")
    folds = sorted([p for p in root.glob("Kfold_*") if p.is_dir()], key=lambda p: p.name)
    if not folds:
        raise FileNotFoundError(f"No Kfold_* under {root}")
    return folds


def _weights_for_fold(train_project_dir: Path, k: int) -> Path:
    # expects Ultralytics layout: <train_project_dir>/k{k}/weights/best.pt
    p = train_project_dir / f"k{k}" / "weights" / "best.pt"
    if not p.exists():
        raise FileNotFoundError(f"Missing weights for fold k{k}: {p}")
    return p


def run_kfold_predictions(cfg: dict):
    """
    cfg keys (with defaults):
      - strategy: "strict" | "smoothed" | "loose"
      - aug_root: "P:/BruiseDet_Repo/data/augmented"
      - train_project_template: "P:/BruiseDet_Repo/outputs/training/{strategy}/yolov9"
      - pred_out_template: "P:/BruiseDet_Repo/outputs/predictions/{strategy}/yolov9"
      - conf: 0.05
      - iou: 0.15
      - imgsz: 640
      - batch: 16
      - ioa_threshold: 0.5
      - device: None  (auto)
    """
    strategy = str(cfg.get("strategy", "strict")).lower()
    strategy_cap = strategy.capitalize()

    aug_root = Path(cfg.get("aug_root", r"P:/BruiseDet_Repo/data/augmented"))
    folds = _collect_folds(aug_root, strategy_cap)

    train_project_dir = Path(
        str(cfg.get("train_project_template", r"P:/BruiseDet_Repo/outputs/training/{strategy}/yolov9"))
        .format(strategy=strategy_cap)
    )
    pred_root = Path(
        str(cfg.get("pred_out_template", r"P:/BruiseDet_Repo/outputs/predictions/{strategy}/yolov9"))
        .format(strategy=strategy_cap)
    )
    pred_root.mkdir(parents=True, exist_ok=True)

    conf = float(cfg.get("conf", 0.05))
    iou = float(cfg.get("iou", 0.15))
    imgsz = int(cfg.get("imgsz", 640))
    batch = int(cfg.get("batch", 16))
    ioa_thr = float(cfg.get("ioa_threshold", 0.5))

    device_cfg = cfg.get("device")
    device = device_cfg if device_cfg else (0 if torch.cuda.is_available() else "cpu")

    for k_idx, kdir in enumerate(folds, start=1):
        print(f"[YOLOv9|pred] fold k{k_idx}")
        weights = _weights_for_fold(train_project_dir, k_idx)

        # load model per fold (keeps memory sane if folds differ)
        model = YOLO(str(weights))

        # val images
        images_dir = kdir / "val" / "images"
        if not images_dir.exists():
            print(f"  [warn] missing val/images under {kdir}, skipping")
            continue
        image_paths = [
            str(p) for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if not image_paths:
            print(f"  [warn] no images in {images_dir}, skipping")
            continue

        # where to write
        out_dir = pred_root / f"k{k_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # batch inference
        results = model.predict(
            image_paths,
            save=False,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            batch=batch,
            device=device
        )

        _save_predictions_for_batch(results, image_paths, out_dir, ioa_threshold=ioa_thr)

    print(f"[YOLOv9|pred] done. predictions -> {pred_root}")
