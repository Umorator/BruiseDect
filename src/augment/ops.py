# src/augment/ops.py
import cv2
import albumentations as A

# -----------------------------
# Bounding-box clamping helpers
# -----------------------------

_EPS = 1e-6

def _clamp01(v: float) -> float:
    """Clamp scalar into [0, 1]."""
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v

def _clamp_yolo_box(x: float, y: float, w: float, h: float, eps: float = _EPS):
    """
    Clamp a YOLO (x, y, w, h) box to [0,1] by clamping corners and
    recomputing center/size. Keeps min size >= eps.
    """
    # clamp center & size
    x = _clamp01(x)
    y = _clamp01(y)
    w = max(eps, min(1.0, w))
    h = max(eps, min(1.0, h))

    # go to corners, clamp corners, then recompute
    x1 = _clamp01(x - w / 2.0)
    y1 = _clamp01(y - h / 2.0)
    x2 = _clamp01(x + w / 2.0)
    y2 = _clamp01(y + h / 2.0)

    # ensure non-degenerate
    if x2 <= x1:
        x2 = min(1.0, x1 + eps)
    if y2 <= y1:
        y2 = min(1.0, y1 + eps)

    w2 = max(eps, x2 - x1)
    h2 = max(eps, y2 - y1)
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    return xc, yc, w2, h2

def _clamp_yolo_list(boxes):
    """
    Clamp a list of YOLO boxes with class:
      boxes: [[x, y, w, h, c], ...]
    Returns a new list with all boxes clamped to [0,1].
    """
    out = []
    for x, y, w, h, c in boxes:
        xc, yc, w2, h2 = _clamp_yolo_box(float(x), float(y), float(w), float(h))
        out.append([xc, yc, w2, h2, int(c)])
    return out


# -----------------------------
# Albumentations pipelines
# -----------------------------

def Augmentation(image, bboxes):
    """
    Augment labeled images.

    Args:
      image: HxWxC BGR (cv2) np.uint8
      bboxes: list of [x, y, w, h, class_id] in YOLO-normalized coords

    Returns:
      (aug_image, aug_bboxes) where aug_bboxes keep YOLO format with class_id
    """
    try:
        aug = A.Compose(
            [
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.RandomScale(scale_limit=0.1, p=0.5),
                A.RandomSizedBBoxSafeCrop(2000, 2000, p=0.5),
                A.Defocus(alias_blur=0.25, p=0.5),
                A.Rotate(limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                min_visibility=0.3,
                label_fields=["class_labels"],  # REQUIRED so Albumentations tracks labels
            ),
        )

        # split coordinates and labels for Albumentations
        bboxes = _clamp_yolo_list(bboxes)  # clamp before aug
        bbs = [b[:4] for b in bboxes]
        labs = [b[4] for b in bboxes]

        out = aug(image=image, bboxes=bbs, class_labels=labs)

        # reattach labels and clamp again to be safe
        aug_boxes = [list(bb) + [lab] for bb, lab in zip(out["bboxes"], out["class_labels"])]
        aug_boxes = _clamp_yolo_list(aug_boxes)

        return out["image"], aug_boxes

    except Exception as e:
        print(f"[Augmentation] Error: {e}")
        return None, None


def SimpleAugmentation(image):
    """
    Augment unlabeled images (no bbox ops).

    Args:
      image: HxWxC BGR (cv2) np.uint8

    Returns:
      aug_image (or None on failure)
    """
    try:
        aug = A.Compose(
            [
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.RandomScale(),
                A.RandomCrop(500, 500, p=0.5),
                A.Defocus(alias_blur=0.25, p=0.5),
                A.Rotate(limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ]
        )
        out = aug(image=image)
        return out["image"]
    except Exception as e:
        print(f"[SimpleAugmentation] Error: {e}")
        return None
