# src/augment/pipeline_ft.py
import os, shutil, multiprocessing, json
from pathlib import Path
from typing import Optional, Tuple, List
import cv2

from .ops import Augmentation, SimpleAugmentation, _clamp_yolo_list
from .mosaics import create_mosaic_from_directory

VALID_EXTS = (".jpg", ".jpeg", ".png")

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _read_yolo_with_class(txt_path: str, default_class=0):
    boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    c, x, y, w, h = parts
                    boxes.append([float(x), float(y), float(w), float(h), int(float(c))])
    return boxes

def _write_yolo_with_class(txt_path: str, boxes):
    with open(txt_path, "w", encoding="utf-8") as f:
        for x, y, w, h, c in boxes:
            f.write(f"{c} {x} {y} {w} {h}\n")

def _resize_min_side(img, size):
    h, w = img.shape[:2]
    if size is None:
        return img
    scale = size / float(min(h, w))
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def _copy_kfold_tree(src_fold: Path, dst_fold: Path):
    # copy train/val trees (images+labels) and keep the same contents
    for sub in ["train", "val"]:
        for leaf in ["images", "labels"]:
            _ensure_dir(dst_fold / sub / leaf)
            src_dir = src_fold / sub / leaf
            if not src_dir.exists():
                continue
            for f in os.listdir(src_dir):
                shutil.copy2(src_dir / f, dst_fold / sub / leaf / f)
    # Create yaml_files folder (we will overwrite later)
    _ensure_dir(dst_fold.parent / "yaml_files")

def _process_extra_one(args):
    img_file, extra_dir, out_train_img, out_train_lbl, times, size, class_id = args
    src_img = os.path.join(extra_dir, "images", img_file)
    src_lbl = os.path.join(extra_dir, "labels", os.path.splitext(img_file)[0] + ".txt")

    img = cv2.imread(src_img)
    if img is None:
        return 0

    # write original (resized if size set)
    img_w = _resize_min_side(img, size)
    base_name = os.path.splitext(img_file)[0]
    cv2.imwrite(os.path.join(out_train_img, img_file), img_w)

    has_label = os.path.exists(src_lbl)
    boxes = []
    if has_label:
        boxes = _read_yolo_with_class(src_lbl, default_class=class_id)
        boxes = _clamp_yolo_list(boxes)
        _write_yolo_with_class(os.path.join(out_train_lbl, base_name + ".txt"), boxes)

    # augment (same policy as earlier pipeline: if labels exist → bbox-aware, else simple)
    if times > 0:
        for i in range(times):
            if has_label and len(boxes) > 0:
                aug_img, aug_boxes = Augmentation(img, boxes)
                if aug_img is None or aug_boxes is None:
                    continue
                aug_name = f"{base_name}_extra_aug_{i}.jpg"
                cv2.imwrite(os.path.join(out_train_img, aug_name), aug_img)
                _write_yolo_with_class(os.path.join(out_train_lbl, f"{base_name}_extra_aug_{i}.txt"), aug_boxes)
            else:
                aug_img = SimpleAugmentation(img)
                if aug_img is None:
                    continue
                aug_name = f"{base_name}_extra_aug_{i}.jpg"
                cv2.imwrite(os.path.join(out_train_img, aug_name), aug_img)
    return 1

def add_extra_to_kfold(
    base_aug_root: str,
    extra_train_dir: str,
    out_aug_ft_root: str,
    times_extra: int = 1,
    size_number: Optional[int] = None,
    resize: bool = False,
    class_id: int = 0,
    mosaic_pct: int = 10,
    yaml_names: Tuple[str, ...] = ("hematoma",),
):
    """
    Clone existing augmented Kfold dataset (for a chosen strategy) and
    ADD an external curated dataset ONLY to TRAIN folds (not val).
    - base_aug_root: e.g. P:/BruiseDet_Repo/data/augmented/Strict
    - extra_train_dir: folder with /images and /labels for curated set (481 imgs)
    - out_aug_ft_root: e.g. P:/BruiseDet_Repo/data/augmented_ft/Strict
    """
    base_aug_root = Path(base_aug_root)
    extra_train_dir = Path(extra_train_dir)
    out_aug_ft_root = Path(out_aug_ft_root)

    _ensure_dir(out_aug_ft_root)
    yaml_dir = out_aug_ft_root / "yaml_files"
    _ensure_dir(yaml_dir)

    folds = sorted([p for p in base_aug_root.glob("Kfold_*") if p.is_dir()], key=lambda x: x.name)
    if not folds:
        raise FileNotFoundError(f"No Kfold_* folders under {base_aug_root}")

    SZ = size_number if resize else None
    nproc = max(1, int(0.75 * multiprocessing.cpu_count()))

    for fold in folds:
        kname = fold.name  # Kfold_1 ...
        out_fold = out_aug_ft_root / kname
        print(f"[finetune] cloning {fold} -> {out_fold}")
        _copy_kfold_tree(fold, out_fold)

        # EXTRA → only into TRAIN
        out_train_img = out_fold / "train" / "images"
        out_train_lbl = out_fold / "train" / "labels"
        _ensure_dir(out_train_img); _ensure_dir(out_train_lbl)

        extra_imgs = [f for f in os.listdir(extra_train_dir / "images") if f.lower().endswith(VALID_EXTS)]
        args = [(f, str(extra_train_dir), str(out_train_img), str(out_train_lbl),
                 int(times_extra), SZ, int(class_id)) for f in extra_imgs]

        if nproc > 1:
            with multiprocessing.Pool(processes=nproc) as pool:
                pool.map(_process_extra_one, args)
        else:
            for a in args:
                _process_extra_one(a)

        # MOSAICS on the merged TRAIN (same percentage)
        print(f"[finetune] mosaics on {out_train_img}")
        create_mosaic_from_directory(
            image_folder=str(out_train_img),
            bbox_folder=str(out_train_lbl),
            output_image_folder=str(out_train_img),
            output_bbox_folder=str(out_train_lbl),
            percentage=int(mosaic_pct),
            default_class_id=int(class_id),
        )

        # Write YOLO data YAML
        train_path = (out_fold / "train" / "images").as_posix()
        val_path   = (out_fold / "val"   / "images").as_posix()
        data = {
            "train": train_path,
            "val": val_path,
            "nc": len(yaml_names),
            "names": list(yaml_names)
        }
        (yaml_dir / f"{kname.lower()}.yaml").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    print(f"[finetune] finished -> {out_aug_ft_root}")
    return str(out_aug_ft_root)
