from pathlib import Path
from typing import List, Dict, Set
import cv2

from .io import read_yolo_txt
from .geometry import (
    iou_xywh, containment, significant_overlap, nms_xywh, xywh_to_xyxy
)

def _list_txt_basenames(folder: Path) -> Set[str]:
    return {
        p.stem for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    }

def find_common_images(folders: List[Path]) -> Set[str]:
    """Return set of image file names with annotations in ALL folders (converted to .jpg names)."""
    if not folders:
        return set()
    common = _list_txt_basenames(folders[0])
    for f in folders[1:]:
        common &= _list_txt_basenames(f)
    # return as canonical image names (jpg)
    return {f"{name}.jpg" for name in common}

def list_all_images(images_roots: List[Path], valid_exts={".jpg", ".jpeg", ".png"}) -> Set[str]:
    names = set()
    for root in images_roots:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in valid_exts:
                names.add(p.name)
    return names

def load_annotations_for_images(folders: List[Path], image_names: Set[str]) -> Dict[str, List[list]]:
    """Return mapping image_name -> [boxes_doc1, boxes_doc2, boxes_doc3]"""
    res = {}
    for img in image_names:
        trio = []
        for f in folders:
            txt_path = f / (Path(img).stem + ".txt")
            trio.append(read_yolo_txt(txt_path))
        res[img] = trio
    return res

def draw_boxes_overlay(image, boxes_list, overlaps):
    """Draw doctors: yellow/green/blue, overlaps red."""
    out = image.copy()
    colors = [(0,255,255),(0,255,0),(255,0,0)]
    red = (0,0,255)

    def to_rect(b, W, H):
        xc, yc, w, h = b
        x1 = int((xc - w/2) * W); y1 = int((yc - h/2) * H)
        x2 = int((xc + w/2) * W); y2 = int((yc + h/2) * H)
        return x1, y1, x2, y2

    H, W = out.shape[:2]
    for i, boxes in enumerate(boxes_list):
        c = colors[i % len(colors)]
        for b in boxes:
            x1,y1,x2,y2 = to_rect(b,W,H)
            cv2.rectangle(out, (x1,y1), (x2,y2), c, 2)
    for b in overlaps:
        x1,y1,x2,y2 = to_rect(b,W,H)
        cv2.rectangle(out, (x1,y1), (x2,y2), red, 2)
    return out
