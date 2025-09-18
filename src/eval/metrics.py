import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# -----------------------------
# Image + box sizing helpers
# -----------------------------

VALID_EXTS = (".jpg", ".jpeg", ".png")


def _find_image(base_dir: Path, stem: str) -> Optional[Path]:
    """Find image by trying common extensions."""
    for ext in VALID_EXTS:
        p = base_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def get_model_resize_dimensions(
    original_size: Tuple[int, int],
    model_type: str,
    target_size: int
) -> Tuple[int, int, float]:
    """
    Compute the processed size (W,H) and scale used to map original absolute pixels to processed absolute pixels.
    - YOLO: longest side -> target_size, short side rounded to nearest multiple of 32 (not exceeding target_size).
    - FRCNN/RetinaNet: longest side -> target_size (simple aspect-preserving scale).
    Returns: (new_w, new_h, scale), where scale = target_size / max(orig_w, orig_h) for both branches,
    but YOLO also snaps the short side to nearest multiple of 32.
    """
    orig_w, orig_h = original_size
    model_type = (model_type or "").lower()

    if model_type == "yolo":
        # scale wrt longest side
        scale = target_size / max(orig_w, orig_h)
        # tentative sizes
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # round both dims to nearest multiple of 32
        def _round32(v: int) -> int:
            return ((v + 16) // 32) * 32 if v > 0 else 32

        new_w = _round32(new_w)
        new_h = _round32(new_h)

        # ensure the longest side equals target_size (post-rounding)
        if orig_w >= orig_h:
            new_w = target_size
            new_h = min(_round32(int(orig_h * (target_size / orig_w))), target_size)
        else:
            new_h = target_size
            new_w = min(_round32(int(orig_w * (target_size / orig_h))), target_size)

        return new_w, new_h, scale

    # FRCNN / RetinaNet
    scale = target_size / max(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    return new_w, new_h, scale


def _adjust_boxes_from_orig_to_processed(
    boxes_xywhn: List[List[float]],
    original_size: Tuple[int, int],
    processed_dims: Tuple[int, int, float],
) -> List[List[float]]:
    """
    Convert normalized boxes relative to the ORIGINAL image size into normalized boxes
    relative to the PROCESSED (resized) image size.
    boxes_xywhn: [[x, y, w, h], ...] normalized to original
    original_size: (orig_w, orig_h)
    processed_dims: (new_w, new_h, scale) where scale = target / max(orig_w, orig_h)
    """
    orig_w, orig_h = original_size
    new_w, new_h, scale = processed_dims
    out = []
    for x, y, w, h in boxes_xywhn:
        # original normalized -> original absolute
        x_abs = x * orig_w
        y_abs = y * orig_h
        w_abs = w * orig_w
        h_abs = h * orig_h
        # scale absolute to processed absolute
        x_proc = x_abs * scale
        y_proc = y_abs * scale
        w_proc = w_abs * scale
        h_proc = h_abs * scale
        # processed absolute -> processed normalized
        out.append([
            x_proc / max(new_w, 1e-9),
            y_proc / max(new_h, 1e-9),
            w_proc / max(new_w, 1e-9),
            h_proc / max(new_h, 1e-9),
        ])
    return out


# -----------------------------
# File IO for labels/preds
# -----------------------------

def read_bounding_boxes_from_txt(txt_file_path: Path) -> List[List[float]]:
    """Read YOLO-normalized boxes 'class x y w h' and return [[x,y,w,h], ...]."""
    boxes = []
    if not txt_file_path.exists():
        return boxes
    with txt_file_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                # class ignored → single class
                _, x, y, w, h = parts
                boxes.append([float(x), float(y), float(w), float(h)])
    return boxes


def load_predictions_with_confidence(file_path: Path) -> Tuple[List[List[float]], List[float]]:
    """
    Read predictions 'class x y w h conf' → (boxes_xywhn, confs).
    Boxes are normalized to either the original or processed size depending on how they were saved.
    """
    boxes, confs = [], []
    if not file_path.exists():
        return boxes, confs
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                _, x, y, w, h, conf = parts[:6]
                boxes.append([float(x), float(y), float(w), float(h)])
                confs.append(float(conf))
    return boxes, confs


# -----------------------------
# Matching + metrics (your logic)
# -----------------------------

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xmin1, ymin1 = x1 - w1 / 2, y1 - h1 / 2
    xmax1, ymax1 = x1 + w1 / 2, y1 + h1 / 2
    xmin2, ymin2 = x2 - w2 / 2, y2 - h2 / 2
    xmax2, ymax2 = x2 + w2 / 2, y2 + h2 / 2

    inter_minx = max(xmin1, xmin2)
    inter_miny = max(ymin1, ymin2)
    inter_maxx = min(xmax1, xmax2)
    inter_maxy = min(ymax1, ymax2)

    inter_area = max(0.0, inter_maxx - inter_minx) * max(0.0, inter_maxy - inter_miny)
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    union = area1 + area2 - inter_area
    return float(inter_area / union) if union > 1e-9 else 0.0


def calculate_dice(box1: List[float], box2: List[float]) -> float:
    iou = calculate_iou(box1, box2)
    return float((2 * iou) / (1 + iou)) if iou > 0 else 0.0


def match_and_calculate_iou_with_confidence(
    image_name: str,
    ground_truth_boxes: List[List[float]],
    predicted_boxes: List[List[float]],
    predicted_confidences: List[float],
    iou_match_threshold: float = 0.5,
) -> Tuple[List[Tuple[str, float, float, float, str]], List[float], List[float]]:
    """
    Greedy per-prediction matching (same as your original):
    - For each prediction, match to GT with maximum IoU.
    - ≥ thresh → TP, else FP.
    - Any unmatched GT → FN.
    """
    output_list = []
    matched_gt_indices = set()
    iou_list, dice_list = [], []

    for pred_idx, (pred_box, conf) in enumerate(zip(predicted_boxes, predicted_confidences)):
        best_iou, best_dice, best_gt = 0.0, 0.0, None
        for gt_idx, gt_box in enumerate(ground_truth_boxes):
            iou = calculate_iou(pred_box, gt_box)
            dice = calculate_dice(pred_box, gt_box)
            if iou > best_iou:
                best_iou, best_dice, best_gt = iou, dice, gt_idx

        if best_iou >= iou_match_threshold and best_gt is not None:
            matched_gt_indices.add(best_gt)
            output_list.append((f"{image_name}_boundingbox_{best_gt}", best_iou, best_dice, conf, "TP"))
            iou_list.append(best_iou)
            dice_list.append(best_dice)
        else:
            output_list.append((f"{image_name}_prediction_{pred_idx}", best_iou, best_dice, conf, "FP"))

    for gt_idx in range(len(ground_truth_boxes)):
        if gt_idx not in matched_gt_indices:
            output_list.append((f"{image_name}_boundingbox_{gt_idx}", 0.0, 0.0, 0.0, "FN"))

    return output_list, iou_list, dice_list


def calculate_total_ground_truth_boxes(folder_path: Path) -> int:
    total = 0
    for p in folder_path.glob("*.txt"):
        total += len(read_bounding_boxes_from_txt(p))
    return total


# -----------------------------
# Single-model evaluation (one strategy, one fold, one model)
# -----------------------------

def evaluate_model(
    gt_folder: Path,
    pred_folder: Path,
    iou_threshold: float,
    resize_number: int,
    model_name: str,
    norm_mode: str,  # "orig" or "resized"
) -> Dict:
    """
    Evaluate one (strategy, fold, model).
    - Boxes are compared in the coordinate system of the PROCESSED (resized) image.
    - GT labels are always YOLO-normalized w.r.t. the ORIGINAL image, so we **always adjust** GT to processed size.
    - Predictions:
        * if norm_mode == "orig": adjust from original -> processed (YOLO style).
        * if norm_mode == "resized": already in processed -> leave as-is.
    - COCO eval is performed in pixel coords using processed width/height per image.
    """
    df = pd.DataFrame(columns=["Type", "IoU", "Dice", "Confidence"])
    total_gt = calculate_total_ground_truth_boxes(gt_folder)
    all_iou, all_dice = [], []

    # Prepare data for COCO evaluation
    coco_gt = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "bruise"}], "info": {}, "licenses": []}
    coco_pred = []

    image_id = 0
    ann_id = 0

    # infer model-type for resizing path ("yolo" vs "frcnn" branch)
    model_type_for_resize = "yolo" if model_name.lower() == "yolo" else "frcnn"

    img_dir = gt_folder.parent / "images"

    for gt_txt in sorted(gt_folder.glob("*.txt")):
        stem = gt_txt.stem
        image_path = _find_image(img_dir, stem)
        if image_path is None:
            # If no image, we still can proceed with a fallback processed size
            original_size = (resize_number, resize_number)
        else:
            with Image.open(image_path) as im:
                original_size = im.size  # (W,H)

        # Compute processed dims exactly as training
        new_w, new_h, scale = get_model_resize_dimensions(original_size, model_type_for_resize, resize_number)
        processed_size = (new_w, new_h)

        image_id += 1
        coco_gt["images"].append({
            "id": image_id,
            "file_name": f"{stem}.jpg",
            "width": processed_size[0],
            "height": processed_size[1],
        })

        # ---- Ground truth (orig -> processed)
        gt_boxes_orig = read_bounding_boxes_from_txt(gt_txt)
        gt_boxes_proc = _adjust_boxes_from_orig_to_processed(gt_boxes_orig, original_size, (new_w, new_h, scale))

        # ---- Predictions
        pred_txt = pred_folder / f"{stem}.txt"
        pred_boxes, pred_confs = load_predictions_with_confidence(pred_txt)

        if norm_mode == "orig":
            # YOLO predictions saved relative to the ORIGINAL size → adjust them
            pred_boxes_proc = _adjust_boxes_from_orig_to_processed(pred_boxes, original_size, (new_w, new_h, scale))
        else:
            # "resized": predictions already normalized to the PROCESSED size
            pred_boxes_proc = pred_boxes

        # ---- Custom matching metrics (boxes in processed-normalized coords)
        matches, iou_list, dice_list = match_and_calculate_iou_with_confidence(
            stem, gt_boxes_proc, pred_boxes_proc, pred_confs, iou_match_threshold=iou_threshold
        )
        all_iou.extend(iou_list)
        all_dice.extend(dice_list)
        for m in matches:
            bbox_name, iou, dice, conf, dtype = m
            df.loc[bbox_name] = [dtype, iou, dice, conf]

        # ---- Add to COCO lists (convert processed-normalized → processed pixels)
        # GT
        for bb in gt_boxes_proc:
            x, y, w, h = bb
            x1 = (x - w / 2) * processed_size[0]
            y1 = (y - h / 2) * processed_size[1]
            bw = w * processed_size[0]
            bh = h * processed_size[1]
            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, bw, bh],
                "area": float(bw * bh),
                "iscrowd": 0,
            })
            ann_id += 1

        # Preds
        for bb, conf in zip(pred_boxes_proc, pred_confs):
            x, y, w, h = bb
            x1 = (x - w / 2) * processed_size[0]
            y1 = (y - h / 2) * processed_size[1]
            bw = w * processed_size[0]
            bh = h * processed_size[1]
            coco_pred.append({
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, bw, bh],
                "score": float(conf),
                "segmentation": [],
            })

    # Aggregate custom metrics
    tp = int((df["Type"] == "TP").sum())
    fp = int((df["Type"] == "FP").sum())
    fn = int((df["Type"] == "FN").sum())

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    avg_iou = float(np.mean(all_iou)) if all_iou else 0.0
    avg_dice = float(np.mean(all_dice)) if all_dice else 0.0

    # COCO AP@0.5
    coco_ap = -1.0
    if len(coco_pred) > 0:
        tmp_gt = Path("temp_gt.json")
        tmp_dt = Path("temp_pred.json")
        tmp_gt.write_text(json.dumps(coco_gt), encoding="utf-8")
        tmp_dt.write_text(json.dumps(coco_pred), encoding="utf-8")
        try:
            cocoGt = COCO(str(tmp_gt))
            cocoDt = cocoGt.loadRes(str(tmp_dt))
            cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
            cocoEval.params.iouThrs = np.array([iou_threshold])  # AP@0.5 only
            cocoEval.evaluate()
            cocoEval.accumulate()
            if "precision" in cocoEval.eval and len(cocoEval.eval["precision"]) > 0:
                prec = cocoEval.eval["precision"][0, :, 0, -1]
                valid = prec[prec > -1]
                coco_ap = float(np.mean(valid)) if valid.size > 0 else -1.0
        except Exception as e:
            print(f"[warn] COCO eval failed: {e}")
        finally:
            try:
                tmp_gt.unlink(missing_ok=True)
                tmp_dt.unlink(missing_ok=True)
            except Exception:
                pass

    metrics = {
        "Model": os.path.basename(str(pred_folder)),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU": avg_iou,
        "Dice": avg_dice,
        "AP": coco_ap,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Total GT": int(total_gt),
        "IOU Threshold": float(iou_threshold),
    }
    return metrics


# -----------------------------
# Orchestrator for (strategy × model × fold)
# -----------------------------

def evaluate_all(cfg: Dict) -> Tuple[str, str]:
    """
    cfg keys:
      strategies: ["Strict","Smoothed","Loose"]
      models: ["YOLO","FRCNN","RetinaNet"]
      aug_root_template: "P:/.../augmented/{strategy}"
      pred_root_template: "P:/.../predictions/{strategy}/{model}"
      out_dir: "P:/.../outputs/eval"
      folds: null | [1,3,5]
      iou_threshold: 0.5
      resize_number: 640
      model_norm: {"YOLO":"orig","FRCNN":"resized","RetinaNet":"resized"}
    """
    strategies = [s.capitalize() for s in cfg.get("strategies", [])]
    models = cfg.get("models", [])
    aug_tpl = cfg["aug_root_template"]
    pred_tpl = cfg["pred_root_template"]
    out_dir = Path(cfg["out_dir"])
    folds = cfg.get("folds")
    folds = folds if folds else [1, 2, 3, 4, 5]
    iou_thr = float(cfg.get("iou_threshold", 0.5))
    resize_number = int(cfg.get("resize_number", 640))
    model_norm = cfg.get("model_norm", {"YOLO": "orig", "FRCNN": "resized", "RetinaNet": "resized"})

    # Run folder
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"eval_run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save specs
    (run_dir / "specs.txt").write_text(
        json.dumps({
            "strategies": strategies,
            "models": models,
            "folds": folds,
            "iou_threshold": iou_thr,
            "resize_number": resize_number,
            "model_norm": model_norm,
            "aug_root_template": aug_tpl,
            "pred_root_template": pred_tpl,
        }, indent=2),
        encoding="utf-8",
    )

    all_rows = []

    for strategy in strategies:
        aug_root = Path(aug_tpl.format(strategy=strategy))
        for model in models:
            norm_mode = (model_norm.get(model, "orig") or "orig").lower()
            for fold in folds:
                gt_dir = aug_root / f"Kfold_{fold}" / "val" / "labels"
                pred_dir = Path(pred_tpl.format(strategy=strategy, model=model)) / f"k{fold}"

                if not gt_dir.exists():
                    print(f"[warn] missing GT: {gt_dir}")
                    continue
                if not pred_dir.exists():
                    print(f"[warn] missing predictions: {pred_dir}")
                    continue

                print(f"[eval] {strategy} | {model} | fold {fold}")
                metrics = evaluate_model(
                    gt_folder=gt_dir,
                    pred_folder=pred_dir,
                    iou_threshold=iou_thr,
                    resize_number=resize_number,
                    model_name=model,
                    norm_mode=norm_mode,
                )
                metrics.update({"Strategy": strategy, "ModelName": model, "Fold": fold})
                all_rows.append(metrics)

    if not all_rows:
        # still write empty CSVs for consistency
        detailed_csv = str(run_dir / "detailed.csv")
        summary_csv = str(run_dir / "summary.csv")
        pd.DataFrame().to_csv(detailed_csv, index=False)
        pd.DataFrame().to_csv(summary_csv, index=False)
        return detailed_csv, summary_csv

    df = pd.DataFrame(all_rows)

    detailed_csv = str(run_dir / "detailed.csv")
    df.to_csv(detailed_csv, index=False)

    # Summary by (Strategy, ModelName)
    metrics_cols = ["Precision", "Recall", "F1", "IoU", "Dice", "AP", "TP", "FP", "FN", "Total GT"]
    grouped = df.groupby(["Strategy", "ModelName"])
    mean_df = grouped[metrics_cols].mean().round(4)
    std_df = grouped[metrics_cols].std().round(4)
    summary = pd.concat({"Mean": mean_df, "Std": std_df}, axis=1).reset_index()

    summary_csv = str(run_dir / "summary.csv")
    summary.to_csv(summary_csv, index=False)
    return detailed_csv, summary_csv
