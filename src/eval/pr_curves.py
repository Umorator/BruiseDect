import json
import math
import tempfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ----------------------------- IO helpers -----------------------------

def _load_cfg(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------- Geometry / resize -----------------------------

def get_model_resize_dimensions(original_size, model_type='yolo', target_size=640):
    """Returns (new_w, new_h, scale) where scale = target/longest_side."""
    orig_w, orig_h = original_size
    if model_type.lower() == "yolo":
        # Longest side -> target_size; short side -> nearest multiple of 32.
        scale = target_size / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        # round to 32
        new_w = (new_w + 16) // 32 * 32
        new_h = (new_h + 16) // 32 * 32
        if orig_w > orig_h:
            new_w = target_size
            new_h = min(int(orig_h * (target_size / orig_w)), target_size)
            new_h = (new_h + 16) // 32 * 32
        else:
            new_h = target_size
            new_w = min(int(orig_w * (target_size / orig_h)), target_size)
            new_w = (new_w + 16) // 32 * 32
        return new_w, new_h, scale
    else:
        # Faster R-CNN / RetinaNet: just longest side -> target_size
        scale = target_size / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        return new_w, new_h, scale

def adjust_boxes_for_resizing(boxes, original_size, resize_info):
    """boxes in normalized xywh (relative to original size); returns normalized xywh on resized canvas."""
    orig_w, orig_h = original_size
    new_w, new_h, scale = resize_info
    out = []
    for x, y, w, h in boxes:
        # absolute
        xa = x * orig_w; ya = y * orig_h
        wa = w * orig_w;  ha = h * orig_h
        # scaled
        xn = (xa * scale) / new_w
        yn = (ya * scale) / new_h
        wn = (wa * scale) / new_w
        hn = (ha * scale) / new_h
        out.append([xn, yn, wn, hn])
    return out


# ----------------------------- Format readers -----------------------------

def read_gt_yolo(txt_path: Path):
    """YOLO normalized xywh; returns [[x,y,w,h], ...]."""
    out = []
    if not txt_path.exists():
        return out
    with txt_path.open("r") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) == 5:
                # class, x, y, w, h
                _, x, y, w, h = parts
                out.append([float(x), float(y), float(w), float(h)])
    return out

def read_pred_xywhn_conf(txt_path: Path):
    """Returns (boxes, confs) where boxes are normalized xywh."""
    boxes, confs = [], []
    if not txt_path.exists():
        return boxes, confs
    with txt_path.open("r") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) >= 6:
                # class, x, y, w, h, conf
                _, x, y, w, h, conf = parts[:6]
                boxes.append([float(x), float(y), float(w), float(h)])
                conffloat = float(conf)
                if conffloat > 1.0:  # Handle potential overflow
                    conffloat = 1.0
                elif conffloat < 0.0:
                    conffloat = 0.0
                confs.append(conffloat)
    return boxes, confs


# ----------------------------- COCO builders -----------------------------

def build_coco_from_dirs(gt_labels_dir: Path,
                         pred_dir: Path,
                         images_exts=(".jpg", ".jpeg", ".png"),
                         norm_policy: str = "orig",
                         target_size: int = 640):
    """
    Build COCO GT+Pred dicts from:
      - gt_labels_dir: .../val/labels
      - pred_dir:      .../predictions/k{fold}
    norm_policy: "orig" (YOLO) or "resized" (FRCNN/RetinaNet)
    """
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "bruise"}],
        "info": {},
        "licenses": []
    }
    coco_pred = []

    img_dir = gt_labels_dir.parent / "images"
    image_id = 0
    ann_id = 0

    for lbl_file in sorted(gt_labels_dir.glob("*.txt")):
        name = lbl_file.stem  # image base name
        # original image path
        img_path = None
        for ext in images_exts:
            p = img_dir / f"{name}{ext}"
            if p.exists():
                img_path = p
                break

        # fallback dimensions if image missing
        if img_path is None:
            orig_size = (target_size, target_size)
        else:
            with Image.open(img_path) as im:
                orig_size = im.size  # (w,h)

        is_yolo = (norm_policy == "orig")
        if is_yolo:
            new_w, new_h, scale = get_model_resize_dimensions(orig_size, "yolo", target_size)
            proc_size = (new_w, new_h)
        else:
            new_w, new_h, scale = get_model_resize_dimensions(orig_size, "frcnn", target_size)
            proc_size = (new_w, new_h)

        # image record
        image_id += 1
        coco_gt["images"].append({
            "id": image_id,
            "file_name": f"{name}.jpg",
            "width": proc_size[0],
            "height": proc_size[1],
        })

        # GT boxes (read from normalized original coords)
        gt_norm = read_gt_yolo(lbl_file)
        if img_path is not None:
            gt_resized = adjust_boxes_for_resizing(gt_norm, orig_size, (new_w, new_h, scale))
        else:
            gt_resized = gt_norm  # assume already in processed coords if no image

        for bb in gt_resized:
            x, y, w, h = bb
            x1 = (x - w/2) * proc_size[0]
            y1 = (y - h/2) * proc_size[1]
            ww = w * proc_size[0]
            hh = h * proc_size[1]
            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, ww, hh],
                "area": float(ww * hh),
                "iscrowd": 0
            })
            ann_id += 1

        # Predictions (normalized)
        pred_file = pred_dir / f"{name}.txt"
        pred_norm, pred_conf = read_pred_xywhn_conf(pred_file)
        if img_path is not None:
            pred_resized = adjust_boxes_for_resizing(pred_norm, orig_size, (new_w, new_h, scale))
        else:
            pred_resized = pred_norm

        for bb, sc in zip(pred_resized, pred_conf):
            x, y, w, h = bb
            x1 = (x - w/2) * proc_size[0]
            y1 = (y - h/2) * proc_size[1]
            ww = w * proc_size[0]
            hh = h * proc_size[1]
            coco_pred.append({
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, ww, hh],
                "score": float(sc),
                "segmentation": []
            })

    return coco_gt, coco_pred


def coco_pr_curve(coco_gt_dict, coco_pred_list, iou_thr=0.5):
    """Return (R, P) arrays from COCOeval at given IoU."""
    with tempfile.TemporaryDirectory() as td:
        gt_p = Path(td) / "gt.json"
        dt_p = Path(td) / "dt.json"
        gt_p.write_text(json.dumps(coco_gt_dict))
        dt_p.write_text(json.dumps(coco_pred_list))

        cocoGt = COCO(str(gt_p))
        cocoDt = cocoGt.loadRes(str(dt_p))
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.params.iouThrs = np.array([iou_thr])
        cocoEval.evaluate()
        cocoEval.accumulate()

        R = cocoEval.params.recThrs  # shape [R]
        # precision dims: [T, R, K, A, M]
        P = cocoEval.eval["precision"][0, :, 0, 0, -1]  # 0th IoU (our single 0.5), class 0, area all, maxDet last
        P = np.where(P < 0, np.nan, P)
        return R, P


# ----------------------------- Plotting -----------------------------

def plot_pr_curves_with_std(curves_data, out_path: Path, title: str = "Precision–Recall"):
    """
    curves_data: list of (label, R, P_mean, P_std) where R,P are 1D arrays on [0,1].
    Shows mean ± 1 std deviation.
    """
    _ensure_dir(out_path.parent)
    plt.figure(figsize=(10, 8))
    
    # Set Seaborn style for better aesthetics
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    
    for label, R, P_mean, P_std in curves_data:
        R = np.asarray(R, dtype=float)
        P_mean = np.asarray(P_mean, dtype=float)
        P_std = np.asarray(P_std, dtype=float)
        
        # Sort by recall for clean plotting
        order = np.argsort(R)
        R_sorted = R[order]
        P_mean_sorted = np.nan_to_num(P_mean[order], nan=0)
        P_std_sorted = np.nan_to_num(P_std[order], nan=0)
        
        # Calculate Average Precision (AP) using trapezoidal integration
        # Remove NaN values for AP calculation
        valid_mask = ~np.isnan(P_mean_sorted) & ~np.isnan(R_sorted)
        if np.any(valid_mask):
            R_valid = R_sorted[valid_mask]
            P_valid = P_mean_sorted[valid_mask]
            # Ensure recall is sorted in ascending order for integration
            sort_idx = np.argsort(R_valid)
            R_sorted_valid = R_valid[sort_idx]
            P_sorted_valid = P_valid[sort_idx]
            # Calculate AP using trapezoidal rule
            ap = np.trapz(P_sorted_valid, R_sorted_valid)
        else:
            ap = 0.0
        
        # Calculate mean standard deviation for the curve
        mean_std = np.nanmean(P_std_sorted)
        
        # Create enhanced label with AP value
        enhanced_label = f"{label} (AP: {ap:.3f} ± {mean_std:.3f})"
        
        # Plot mean line
        plt.plot(R_sorted, P_mean_sorted, label=enhanced_label, linewidth=2.5)
        
        # Plot confidence interval (mean ± 1 std)
        plt.fill_between(R_sorted, 
                        np.clip(P_mean_sorted - P_std_sorted, 0, 1),
                        np.clip(P_mean_sorted + P_std_sorted, 0, 1),
                        alpha=0.2)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"{title} (Mean ± 1 SD)", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pr_curves_all_strategies_with_std(curves_data, out_path: Path, title: str = "Precision–Recall"):
    """
    Special plotting function for ALL strategies with model-based coloring and strategy-based linestyles.
    curves_data: list of (label, R, P_mean, P_std) where label format is "Model (Strategy)"
    """
    _ensure_dir(out_path.parent)
    plt.figure(figsize=(12, 8))
    
    # Set Seaborn style for better aesthetics
    sns.set_style("whitegrid")
    
    # Define color palette for models and linestyles for strategies
    model_colors = {
        'yolov9': '#1f77b4',      # blue
        'FRCNN': '#ff7f0e',     # orange
        'RetinaNet': '#2ca02c'  # green
    }
    
    strategy_styles = {
        'Strict': '-',          # solid
        'Smoothed': '--',       # dashed
        'Loose': ':',           # dotted
        'Original': '-.'        # dash-dot
    }
    
    for label, R, P_mean, P_std in curves_data:
        # Parse label to extract model and strategy
        if ' (' in label and ')' in label:
            model = label.split(' (')[0]
            strategy = label.split(' (')[1].rstrip(')')
        else:
            model = label
            strategy = "Unknown"
        
        # Get color and linestyle
        color = model_colors.get(model, '#000000')  # black for unknown models
        linestyle = strategy_styles.get(strategy, '-')  # solid for unknown strategies
        
        R = np.asarray(R, dtype=float)
        P_mean = np.asarray(P_mean, dtype=float)
        P_std = np.asarray(P_std, dtype=float)
        
        order = np.argsort(R)
        R_sorted = R[order]
        P_mean_sorted = np.nan_to_num(P_mean[order], nan=0)
        P_std_sorted = np.nan_to_num(P_std[order], nan=0)
        
        # Calculate Average Precision (AP) using trapezoidal integration
        # Remove NaN values for AP calculation
        valid_mask = ~np.isnan(P_mean_sorted) & ~np.isnan(R_sorted)
        if np.any(valid_mask):
            R_valid = R_sorted[valid_mask]
            P_valid = P_mean_sorted[valid_mask]
            # Ensure recall is sorted in ascending order for integration
            sort_idx = np.argsort(R_valid)
            R_sorted_valid = R_valid[sort_idx]
            P_sorted_valid = P_valid[sort_idx]
            # Calculate AP using trapezoidal rule
            ap = np.trapz(P_sorted_valid, R_sorted_valid)
        else:
            ap = 0.0
        
        # Calculate mean standard deviation for the curve
        mean_std = np.nanmean(P_std_sorted)
        
        # Create enhanced label with AP value
        enhanced_label = f"{label} (AP: {ap:.3f} ± {mean_std:.3f})"
        
        # Plot mean line
        plt.plot(R_sorted, P_mean_sorted, label=enhanced_label, linewidth=2.5, 
                color=color, linestyle=linestyle)
        
        # Plot confidence interval with matching color but no linestyle
        plt.fill_between(R_sorted,
                        np.clip(P_mean_sorted - P_std_sorted, 0, 1),
                        np.clip(P_mean_sorted + P_std_sorted, 0, 1),
                        alpha=0.2, color=color)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"{title} (Mean ± 1 SD)", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()