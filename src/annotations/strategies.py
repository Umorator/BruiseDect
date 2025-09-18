from typing import List
from .geometry import iou_xywh, containment, nms_xywh

def _pget(params, key, default=None):
    # params is a SimpleNamespace; this also works if you ever pass a dict
    return getattr(params, key, default) if not isinstance(params, dict) else params.get(key, default)

def _pairwise_overlaps(b1, b2, iou_thr, cont_thr, min_ratio):
    overlaps = []
    for a in b1:
        for b in b2:
            iou, inter, a1, a2, (x1,y1,x2,y2) = iou_xywh(a, b)
            cond = (
                (iou >= iou_thr) or
                containment(a, b, thr=cont_thr) or
                containment(b, a, thr=cont_thr)
            )
            if not cond or not (x2 > x1 and y2 > y1):
                continue
            w, h = (x2 - x1), (y2 - y1)
            smaller = min(a1, a2)
            if smaller > 0 and (w*h) >= 0.10*smaller and (w*h)/smaller >= min_ratio:
                xc, yc = (x1 + x2)/2, (y1 + y2)/2
                overlaps.append([xc, yc, w, h])
    return overlaps

def strict_strategy(boxes_list, params):
    if len(boxes_list) != 3:
        raise ValueError("Expected boxes from three doctors")

    iou_thr   = _pget(params, "iou_threshold")
    cont_thr  = _pget(params, "containment_threshold", 0.8)
    min_ratio = _pget(params, "min_overlap_ratio", 0.20)
    nms_iou   = _pget(params, "nms_iou", 0.20)

    b0, b1, b2 = boxes_list
    o01   = _pairwise_overlaps(b0, b1, iou_thr, cont_thr, min_ratio)
    final = _pairwise_overlaps(o01, b2, iou_thr, cont_thr, min_ratio)
    return nms_xywh(final, iou_thr=nms_iou, small_ratio_thr=0.20)

def smoothed_strategy(boxes_list: List[List[list]], params):
    if len(boxes_list) != 3:
        raise ValueError("Expected boxes from three doctors")

    iou_thr   = _pget(params, "iou_threshold")
    min_ratio = _pget(params, "min_overlap_ratio", 0.20)
    nms_iou   = _pget(params, "nms_iou", 0.20)

    overlaps = []
    for i in range(3):
        for j in range(i+1, 3):
            for b1 in boxes_list[i]:
                for b2 in boxes_list[j]:
                    iou, inter, a1, a2, (x1,y1,x2,y2) = iou_xywh(b1, b2)
                    if iou >= iou_thr and (x2 > x1 and y2 > y1):
                        area_overlap = (x2 - x1) * (y2 - y1)
                        smaller = min(a1, a2)
                        if smaller > 0 and (area_overlap / smaller) >= min_ratio:
                            overlaps.append([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])

    return nms_xywh(overlaps, iou_thr=nms_iou, small_ratio_thr=0.20)

def loose_strategy(boxes_list: List[List[list]], params):
    iou_thr  = _pget(params, "iou_threshold")
    full_ct  = _pget(params, "containment_threshold_full", 1.0)
    nms_iou  = _pget(params, "nms_iou", 0.20)

    all_boxes = [b for doc in boxes_list for b in doc]
    overlapping, non_overlapping = [], []

    for i, box in enumerate(all_boxes):
        hit = False
        for j, other in enumerate(all_boxes):
            if i == j: 
                continue
            iou, _, _, _, _ = iou_xywh(box, other)
            if (iou >= iou_thr) or containment(box, other, thr=full_ct):
                hit = True
                break
        (overlapping if hit else non_overlapping).append(box)

    merged = nms_xywh(overlapping, iou_thr=nms_iou, small_ratio_thr=0.20)
    return non_overlapping + merged

def get_strategy(name: str):
    name = name.lower()
    if name == "strict":   return strict_strategy
    if name == "smoothed": return smoothed_strategy
    if name == "loose":    return loose_strategy
    raise ValueError(f"Unknown strategy: {name}")
