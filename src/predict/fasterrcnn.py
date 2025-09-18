import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import nms


def _cap_strategy(s: str) -> str:
    return str(s).strip().lower().capitalize()


def _resize_max_side(image: Image.Image, resize_number: int = 640):
    w, h = image.size
    max_side = max(w, h)
    if max_side == 0:
        return image, 1.0, (w, h)
    r = resize_number / max_side
    new_w, new_h = int(w * r), int(h * r)
    return image.resize((new_w, new_h), Image.LANCZOS), r, (new_w, new_h)


def _ioa_xyxy(a, b) -> float:
    """IOA between two xyxy boxes in pixel coordinates."""
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    denom = min(area_a, area_b)
    return inter / denom if denom > 0 else 0.0


def _list_images(img_dir: Path) -> List[Path]:
    if not img_dir.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}
    return [p for p in img_dir.iterdir() if p.suffix in exts and not p.name.startswith(".")]


def _load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    # Model trained in repo uses num_classes=2 (background + hematoma)
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state)  # strict=True by default; matches your training head
    model.to(device).eval()
    return model


@torch.no_grad()
def _predict_one(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    *,
    score_threshold: float,
    nms_thresh: float,
    ioa_threshold: float,
    resize_number: int,
) -> List[str]:
    """
    Returns list of lines: 'class x y w h conf' (normalized to resized dims).
    """
    img = Image.open(str(image_path)).convert("RGB")
    img_resized, ratio, (W, H) = _resize_max_side(img, resize_number)
    x = to_tensor(img_resized).unsqueeze(0).to(device)

    pred = model(x)[0]
    if len(pred["boxes"]) == 0:
        return []

    keep_nms = nms(pred["boxes"], pred["scores"], nms_thresh)
    boxes = pred["boxes"][keep_nms].cpu().tolist()
    scores = pred["scores"][keep_nms].cpu().tolist()

    # IOA de-dup (pairwise against already kept)
    keep = []
    for i, b in enumerate(boxes):
        drop = False
        for j in keep:
            if _ioa_xyxy(b, boxes[j]) > ioa_threshold:
                drop = True
                break
        if not drop:
            keep.append(i)

    lines = []
    for i in keep:
        if scores[i] < score_threshold:
            continue
        xmin, ymin, xmax, ymax = boxes[i]
        xc = ((xmin + xmax) / 2) / W
        yc = ((ymin + ymax) / 2) / H
        bw = (xmax - xmin) / W
        bh = (ymax - ymin) / H
        # single class -> '0'
        lines.append(f"0 {xc} {yc} {bw} {bh} {scores[i]}")
    return lines


def predict_kfold(cfg: Dict) -> Dict[str, int]:
    """
    cfg:
      strategy: "strict" | "smoothed" | "loose"
      weights_dir_template: "P:/BruiseDet_Repo/outputs/training/{strategy}/FRCNN"
      aug_data_root:        "P:/BruiseDet_Repo/data/augmented/{strategy}"
      out_dir_template:     "P:/BruiseDet_Repo/outputs/predictions/{strategy}/FRCNN"
      folds: null or [1,2,...]

      score_threshold: 0.05
      nms: 0.15
      ioa_threshold: 0.5
      resize_number: 640
      device: null   # "0" or "cpu"
    """
    strat = _cap_strategy(cfg.get("strategy", "strict"))
    weights_root = Path(cfg.get("weights_dir_template", r"P:/BruiseDet_Repo/outputs/training/{strategy}/FRCNN").format(strategy=strat))
    aug_root     = Path(cfg.get("aug_data_root", r"P:/BruiseDet_Repo/data/augmented/{strategy}").format(strategy=strat))
    out_root     = Path(cfg.get("out_dir_template", r"P:/BruiseDet_Repo/outputs/predictions/{strategy}/FRCNN").format(strategy=strat))

    score_thr = float(cfg.get("score_threshold", 0.05))
    nms_thr   = float(cfg.get("nms", 0.15))
    ioa_thr   = float(cfg.get("ioa_threshold", 0.5))
    resize_nb = int(cfg.get("resize_number", 640))
    dev_cfg   = cfg.get("device", None)

    # device
    if dev_cfg is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        # allow "0" -> cuda:0
        if str(dev_cfg).lower() == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{dev_cfg}") if torch.cuda.is_available() else torch.device("cpu")

    # folds
    if cfg.get("folds"):
        folds = [int(k) for k in cfg["folds"]]
    else:
        kdirs = sorted([p for p in aug_root.glob("Kfold_*") if p.is_dir()], key=lambda p: p.name)
        folds = [int(p.name.split("_")[-1]) for p in kdirs]

    if not folds:
        raise FileNotFoundError(f"No Kfold_* found under: {aug_root}")

    print(f"[frcnn|predict] strategy={strat}")
    print(f"[frcnn|predict] weights: {weights_root}")
    print(f"[frcnn|predict] val data root: {aug_root}")
    print(f"[frcnn|predict] out: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    per_fold_counts: Dict[str, int] = {}

    for k in folds:
        wpath = weights_root / f"k{k}" / "best_model.pth"
        img_dir = aug_root / f"Kfold_{k}" / "val" / "images"
        if not wpath.exists():
            print(f"[warn] missing weights for k{k}: {wpath} (skip)")
            continue
        imgs = _list_images(img_dir)
        if not imgs:
            print(f"[warn] no images under: {img_dir} (skip)")
            continue

        save_dir = out_root / f"k{k}"
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"[frcnn|predict] k{k}: {len(imgs)} images | weights={wpath}")
        model = _load_model(wpath, device)

        written = 0
        for ip in imgs:
            lines = _predict_one(
                model, ip, device,
                score_threshold=score_thr,
                nms_thresh=nms_thr,
                ioa_threshold=ioa_thr,
                resize_number=resize_nb
            )
            out_txt = save_dir / (Path(ip).stem + ".txt")
            with out_txt.open("w") as f:
                for ln in lines:
                    f.write(ln + "\n")
            written += 1

        per_fold_counts[f"k{k}"] = written
        print(f"[frcnn|predict] k{k}: wrote {written} files -> {save_dir}")

    return per_fold_counts
