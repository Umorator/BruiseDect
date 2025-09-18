import os, cv2, shutil, multiprocessing, json, random
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
import yaml

from .ops import Augmentation, SimpleAugmentation, _clamp_yolo_list

# ----- YAML inline list helper -----
class _FlowList(list):
    """Dump YAML sequence in flow style: [a, b, c]"""
    pass

def _flow_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(_FlowList, _flow_list_representer)

VALID_EXTS = (".jpg", ".jpeg", ".png")

# ----- tiny utils -----
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_yolo_with_class(txt_path, default_class=0):
    boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    c, x, y, w, h = parts
                    boxes.append([float(x), float(y), float(w), float(h), int(float(c))])
    return boxes

def write_yolo_with_class(txt_path, boxes):
    with open(txt_path, "w", encoding="utf-8") as f:
        for x, y, w, h, c in boxes:
            f.write(f"{c} {x} {y} {w} {h}\n")

def resize_min_side(img, size):
    h, w = img.shape[:2]
    scale = size / float(min(h, w))
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def _normalize_extra_dirs(extra):
    if not extra:
        return []
    if isinstance(extra, (list, tuple)):
        return list(extra)
    return [extra]

def _build_staging_dir(primary_dir, extra_dirs, out_base, collision_policy="skip"):
    """
    Create a flat _staging folder under out_base that contains:
      - all images from primary_dir
      - all images from each extra dir
      - copy matching .txt labels when present
    Skip duplicates by filename (default).
    """
    staging = os.path.join(out_base, "_staging")
    ensure_dir(Path(staging))

    def _copy_one(src_dir, name):
        src_img = os.path.join(src_dir, name)
        dst_img = os.path.join(staging, name)
        if os.path.exists(dst_img):
            if collision_policy == "skip":
                return False
        shutil.copy2(src_img, dst_img)
        # label
        src_lbl = os.path.join(src_dir, os.path.splitext(name)[0] + ".txt")
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, os.path.join(staging, os.path.splitext(name)[0] + ".txt"))
        return True

    def _ingest_dir(src_dir):
        cnt = 0
        for f in os.listdir(src_dir):
            if f.lower().endswith(VALID_EXTS):
                try:
                    ok = _copy_one(src_dir, f)
                    if ok:
                        cnt += 1
                except Exception as e:
                    print(f"[staging] skip {f}: {e}")
        return cnt

    total = 0
    total += _ingest_dir(primary_dir)
    for d in _normalize_extra_dirs(extra_dirs):
        if d and os.path.isdir(d):
            total += _ingest_dir(d)

    print(f"[staging] staged {total} images into: {staging}")
    return staging

# ----- split spec helpers -----
def _save_master_split(path: Path, files_per_fold: dict, all_files: list, kfold: int, seed: int):
    obj = {
        "kfold": int(kfold),
        "seed": int(seed),
        "all_files": list(all_files),          # sorted basenames
        "files_per_fold": files_per_fold       # {"1":{"train":[...],"val":[...]}, ...}
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"[split] master split spec saved -> {path}")

def _load_master_split(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj["files_per_fold"]

# ----- worker -----
def _process_one(args):
    file, input_dir, out_base, split, times, size, class_id = args
    src_img = os.path.join(input_dir, file)
    src_lbl = os.path.join(input_dir, os.path.splitext(file)[0] + ".txt")

    out_img_dir = os.path.join(out_base, split, "images") if split else os.path.join(out_base, "images")
    out_lbl_dir = os.path.join(out_base, split, "labels") if split else os.path.join(out_base, "labels")
    ensure_dir(Path(out_img_dir)); ensure_dir(Path(out_lbl_dir))

    img = cv2.imread(src_img)
    if img is None:
        return 0

    if size:
        img = resize_min_side(img, size)

    dst_img = os.path.join(out_img_dir, file)
    cv2.imwrite(dst_img, img)

    has_label = os.path.exists(src_lbl)
    boxes = []
    if has_label:
        boxes = read_yolo_with_class(src_lbl, default_class=class_id)
        boxes = _clamp_yolo_list(boxes)
        # copy original labels (normalized YOLO → unaffected by resizing)
        write_yolo_with_class(
            os.path.join(out_lbl_dir, os.path.splitext(file)[0] + ".txt"),
            boxes
        )

    # augment only for train or when not splitting
    if (split == "train" or split is None) and times > 0:
        if has_label and len(boxes) > 0:
            for i in range(times):
                aug_img, aug_boxes = Augmentation(img, boxes)
                if aug_img is None or aug_boxes is None:
                    continue
                name = f"{os.path.splitext(file)[0]}_aug_{i}.jpg"
                cv2.imwrite(os.path.join(out_img_dir, name), aug_img)
                with open(os.path.join(out_lbl_dir, f"{os.path.splitext(file)[0]}_aug_{i}.txt"), "w", encoding="utf-8") as f:
                    for x, y, w, h, c in aug_boxes:
                        f.write(f"{c} {x} {y} {w} {h}\n")
        else:
            for i in range(times):
                aug_img = SimpleAugmentation(img)
                if aug_img is None:
                    continue
                name = f"{os.path.splitext(file)[0]}_aug_{i}.jpg"
                cv2.imwrite(os.path.join(out_img_dir, name), aug_img)
    return 1

# ----- public API -----
def _resolve_out_base(input_dir, output_dir_name):
    # If output_dir_name is absolute, use it directly; else join under input_dir
    return output_dir_name if os.path.isabs(output_dir_name) else os.path.join(input_dir, output_dir_name)

def split_and_move_data(
    input_dir, output_dir_name, ratios, times,
    size_number=None, resize=False, class_id=0, mosaic_pct=10,
    extra_images_dir=None, keep_staging=False
):
    """Split primary (+ optional extra) images into train/val, augment, make mosaics."""
    assert abs(sum(ratios) - 1.0) < 1e-8, "ratios must sum to 1.0"
    out_base = _resolve_out_base(input_dir, output_dir_name)
    ensure_dir(Path(out_base))
    for sub in ["train", "val"]:
        ensure_dir(Path(out_base) / sub / "images")
        ensure_dir(Path(out_base) / sub / "labels")

    # Stage combined source (primary + extras)
    src_dir = _build_staging_dir(input_dir, extra_images_dir, out_base)
    files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(VALID_EXTS)])

    if ratios[1] == 0:
        train_files, val_files = files, []
    else:
        train_files, val_files = train_test_split(files, test_size=ratios[1], random_state=123)

    nproc = max(1, int(0.75 * multiprocessing.cpu_count()))
    SZ = size_number if resize else None

    args_train = [(f, src_dir, out_base, "train", times, SZ, class_id) for f in train_files]
    args_val   = [(f, src_dir, out_base, "val",   0,    SZ, class_id) for f in val_files]

    with multiprocessing.Pool(processes=nproc) as pool:
        pool.map(_process_one, args_train)
    for a in args_val:
        _process_one(a)

    # mosaics on training set
    from .mosaics import create_mosaic_from_directory
    create_mosaic_from_directory(
        image_folder=os.path.join(out_base, "train", "images"),
        bbox_folder=os.path.join(out_base, "train", "labels"),
        output_image_folder=os.path.join(out_base, "train", "images"),
        output_bbox_folder=os.path.join(out_base, "train", "labels"),
        percentage=mosaic_pct,
        default_class_id=class_id,
        seed=123  # static seed for non-kfold mode
    )

    # cleanup _staging
    try:
        if not keep_staging and os.path.basename(src_dir) == "_staging":
            shutil.rmtree(src_dir, ignore_errors=True)
    except Exception as e:
        print(f"[cleanup] failed to remove staging: {e}")

    return out_base

def augment_entire_dataset(
    input_dir, output_dir_name, times,
    size_number=None, resize=False, class_id=0, mosaic_pct=10,
    extra_images_dir=None, keep_staging=False
):
    """Augment every image (no split)."""
    out_base = _resolve_out_base(input_dir, output_dir_name)
    ensure_dir(Path(out_base) / "images")
    ensure_dir(Path(out_base) / "labels")

    # Stage combined source
    src_dir = _build_staging_dir(input_dir, extra_images_dir, out_base)
    files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(VALID_EXTS)])

    nproc = max(1, int(0.75 * multiprocessing.cpu_count()))
    SZ = size_number if resize else None

    args_all = [(f, src_dir, out_base, None, times, SZ, class_id) for f in files]
    with multiprocessing.Pool(processes=nproc) as pool:
        pool.map(_process_one, args_all)

    from .mosaics import create_mosaic_from_directory
    create_mosaic_from_directory(
        image_folder=os.path.join(out_base, "images"),
        bbox_folder=os.path.join(out_base, "labels"),
        output_image_folder=os.path.join(out_base, "images"),
        output_bbox_folder=os.path.join(out_base, "labels"),
        percentage=mosaic_pct,
        default_class_id=class_id,
        seed=123
    )

    # cleanup _staging
    try:
        if not keep_staging and os.path.basename(src_dir) == "_staging":
            shutil.rmtree(src_dir, ignore_errors=True)
    except Exception as e:
        print(f"[cleanup] failed to remove staging: {e}")

    return out_base

def split_and_move_data_kfold(
    input_dir, output_dir_name, kfold, times,
    size_number=None, resize=False, class_id=0, mosaic_pct=10,
    yaml_names=("hematoma",), extra_images_dir=None, keep_staging=False,
    split_spec: str | None = None,
    mosaic_seed_base: int = 12345
):
    """
    K-fold split on (primary + optional extra) images, augment per fold, make mosaics, write YOLO YAMLs.
    If split_spec is provided:
      - If it exists, load filenames per fold and use them.
      - If it doesn't exist, create a master split spec from the staged file list, save, then use it.
    """
    assert kfold > 0
    out_base = _resolve_out_base(input_dir, output_dir_name)
    ensure_dir(Path(out_base))

    # Stage combined source once (the staged images define the universe for the split)
    src_dir = _build_staging_dir(input_dir, extra_images_dir, out_base)
    files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(VALID_EXTS)])

    # Build or load the master split spec
    files_per_fold = None
    if split_spec:
        spec_path = Path(split_spec)
        if spec_path.exists():
            print(f"[split] loading master split spec: {spec_path}")
            files_per_fold = _load_master_split(spec_path)
        else:
            print(f"[split] creating master split spec: {spec_path}")
            # Deterministic KFold over sorted basenames
            kf = KFold(n_splits=kfold, shuffle=True, random_state=123)
            files_per_fold = {}
            idx = list(range(len(files)))
            for fold, (tr_idx, va_idx) in enumerate(kf.split(idx), start=1):
                train_files = [files[i] for i in tr_idx]
                val_files   = [files[i] for i in va_idx]
                files_per_fold[str(fold)] = {"train": train_files, "val": val_files}
            _save_master_split(spec_path, files_per_fold, files, kfold, seed=123)
    else:
        # Default deterministic split if no spec path is provided
        kf = KFold(n_splits=kfold, shuffle=True, random_state=123)
        files_per_fold = {}
        idx = list(range(len(files)))
        for fold, (tr_idx, va_idx) in enumerate(kf.split(idx), start=1):
            train_files = [files[i] for i in tr_idx]
            val_files   = [files[i] for i in va_idx]
            files_per_fold[str(fold)] = {"train": train_files, "val": val_files}

    # Process each fold according to the (possibly loaded) spec
    nproc = max(1, int(0.75 * multiprocessing.cpu_count()))
    SZ = size_number if resize else None
    yaml_dir = os.path.join(out_base, "yaml_files")
    ensure_dir(Path(yaml_dir))

    # For quick membership checks in case some images are missing in this strategy
    avail = set(files)

    for fold_str, split_dict in files_per_fold.items():
        fold = int(fold_str)
        fold_dir = os.path.join(out_base, f"Kfold_{fold}")
        for sub in ["train", "val"]:
            ensure_dir(Path(fold_dir) / sub / "images")
            ensure_dir(Path(fold_dir) / sub / "labels")

        # Filter to staged availability (warn on misses)
        spec_train = split_dict.get("train", [])
        spec_val   = split_dict.get("val", [])
        missed_train = [f for f in spec_train if f not in avail]
        missed_val   = [f for f in spec_val if f not in avail]
        if missed_train or missed_val:
            print(f"[warn][fold {fold}] {len(missed_train)} train / {len(missed_val)} val files from spec "
                  f"not found in staged set for this strategy. They will be skipped.")

        train_files = [f for f in spec_train if f in avail]
        val_files   = [f for f in spec_val if f in avail]

        args_train = [(f, src_dir, fold_dir, "train", times, SZ, class_id) for f in train_files]
        args_val   = [(f, src_dir, fold_dir, "val",   0,    SZ, class_id) for f in val_files]

        with multiprocessing.Pool(processes=nproc) as pool:
            pool.map(_process_one, args_train)
        for a in args_val:
            _process_one(a)

        # mosaics on training with deterministic seed per fold
        from .mosaics import create_mosaic_from_directory
        mosaic_seed = int(f"{mosaic_seed_base}{fold}")  # simple derivation
        create_mosaic_from_directory(
            image_folder=os.path.join(fold_dir, "train", "images"),
            bbox_folder=os.path.join(fold_dir, "train", "labels"),
            output_image_folder=os.path.join(fold_dir, "train", "images"),
            output_bbox_folder=os.path.join(fold_dir, "train", "labels"),
            percentage=mosaic_pct,
            default_class_id=class_id,
            seed=mosaic_seed
        )

        # Write YOLO data YAML (POSIX paths + inline names)
        train_path = Path(fold_dir, "train", "images").as_posix()
        val_path   = Path(fold_dir, "val", "images").as_posix()
        data = {
            "train": train_path,
            "val":   val_path,
            "nc":    len(yaml_names),
            "names": _FlowList(list(yaml_names)),  # inline flow style: [hematoma]
        }
        yaml_path = Path(yaml_dir, f"kfold_{fold}.yaml")
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False)

    # cleanup _staging
    try:
        if not keep_staging and os.path.basename(src_dir) == "_staging":
            shutil.rmtree(src_dir, ignore_errors=True)
    except Exception as e:
        print(f"[cleanup] failed to remove staging: {e}")

    return out_base
