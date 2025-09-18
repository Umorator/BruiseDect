import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import retinanet_resnet50_fpn

# tvrefs (vendor'ed torchvision references)
from src.train.tvrefs import utils
from src.train.tvrefs.engine import train_one_epoch as tv_train_one_epoch, evaluate as tv_evaluate


# ----------------------------- IO / dataset helpers -----------------------------

def read_yolo_annotations(txt_file_path: str, img_width: int, img_height: int):
    """YOLO txt (class x y w h normalized) -> pixel xyxy + labels + areas"""
    boxes, areas, labels = [], [], []
    if not os.path.exists(txt_file_path):
        return (torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros(0, dtype=torch.int64),
                torch.zeros(0, dtype=torch.float32))

    with open(txt_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            x_pix, y_pix = x * img_width, y * img_height
            w_pix, h_pix = w * img_width, h * img_height
            xmin = x_pix - w_pix / 2
            ymin = y_pix - h_pix / 2
            xmax = x_pix + w_pix / 2
            ymax = y_pix + h_pix / 2
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                areas.append((xmax - xmin) * (ymax - ymin))
                labels.append(1)  # single foreground class

    boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)
    areas_t = torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros(0, dtype=torch.float32)
    return boxes_t, labels_t, areas_t


class IMPACT_AI(torch.utils.data.Dataset):
    """
    Expects:
      <root_dir>/
        images/
        labels/
    with YOLO txt labels (class x y w h, normalized).
    """
    def __init__(self, root_dir: str, resize_number: int = 640, cocoval: bool = False):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "images")
        self.lbl_dir = os.path.join(root_dir, "labels")
        self.imgs = sorted([f for f in os.listdir(self.img_dir)
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        self.resize_number = int(resize_number)
        self.cocoval = bool(cocoval)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        lbl_path = os.path.join(self.lbl_dir, os.path.splitext(img_name)[0] + ".txt")

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        boxes, labels, areas = read_yolo_annotations(lbl_path, w, h)

        # resize keeping aspect ratio (longest side -> resize_number)
        max_side = max(w, h)
        ratio = self.resize_number / max(max_side, 1)
        new_w, new_h = int(w * ratio), int(h * ratio)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= ratio
            boxes[:, [1, 3]] *= ratio

        image = to_tensor(image)
        num_objs = boxes.shape[0]
        iscrowd = torch.zeros((num_objs,), dtype=torch.uint8)
        image_id = idx if self.cocoval else torch.tensor([idx])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd,
        }
        return image, target


# ----------------------------- RetinaNet model helpers -----------------------------
def initialize_retinanet(num_classes: int,
                         pretrained: bool = True,
                         trainable_backbone_layers: int = 0,
                         dropout_prob: float | None = 0.3,
                         weights_path: str | None = None):
    """
    Build RetinaNet with optional pre-trained weights.
    """
    if weights_path and os.path.exists(weights_path):
        print(f"Loading pre-trained weights from: {weights_path}")
        
        # First, let's inspect the pre-trained model to understand its architecture
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Check if this is a checkpoint file or direct model weights
        if 'model_state' in state_dict:
            # It's a checkpoint file, extract the model state
            model_state = state_dict['model_state']
            print("Loading from checkpoint file")
        else:
            # It's direct model weights
            model_state = state_dict
            print("Loading from model weights file")
        
        # Analyze the classification head to determine the original number of classes
        cls_logits_weight_key = None
        for key in model_state.keys():
            if 'cls_logits.weight' in key:
                cls_logits_weight_key = key
                break
        
        if cls_logits_weight_key:
            # Calculate original number of classes based on weight shape
            original_weight_shape = model_state[cls_logits_weight_key].shape
            # Shape is [num_anchors * num_classes, 256, 3, 3]
            original_num_classes = original_weight_shape[0] // 9  # Assuming 9 anchors
            print(f"Pre-trained model was trained for {original_num_classes} classes")
        else:
            # Couldn't find classification head, assume default COCO
            original_num_classes = 91
            print("Could not determine original number of classes, assuming COCO (91)")
        
        # Create model with the same architecture as the pre-trained model
        model = retinanet_resnet50_fpn(weights=None,
                                     trainable_backbone_layers=trainable_backbone_layers)
        
        # Set up the classification head to match the pre-trained model
        in_channels = model.head.classification_head.conv[0][0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head.cls_logits = nn.Conv2d(
            in_channels, num_anchors * original_num_classes, kernel_size=3, stride=1, padding=1
        )
        model.head.classification_head.num_classes = original_num_classes
        
        # Load the complete state dict
        model.load_state_dict(model_state, strict=True)  # Use strict=True to catch mismatches
        
        # Now if we need a different number of classes, replace just the classification head
        if num_classes != original_num_classes:
            print(f"Replacing classification head from {original_num_classes} to {num_classes} classes")
            model.head.classification_head.cls_logits = nn.Conv2d(
                in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
            )
            model.head.classification_head.num_classes = num_classes
        
        # Optional: add dropout
        if dropout_prob is not None:
            class ConvReluDrop(nn.Sequential):
                def __init__(self, in_c, out_c, p):
                    super().__init__(
                        nn.Conv2d(in_c, out_c, 3, 1, 1),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p)
                    )
            conv_seq = list(model.head.classification_head.conv.children())
            last = conv_seq[-1][0]
            conv_seq[-1] = ConvReluDrop(last.in_channels, last.out_channels, float(dropout_prob))
            model.head.classification_head.conv = nn.Sequential(*conv_seq)
        
        print("Successfully loaded pre-trained weights")
        
    else:
        # Default initialization
        weights = "COCO_V1" if pretrained else None
        model = retinanet_resnet50_fpn(weights=weights,
                                       trainable_backbone_layers=trainable_backbone_layers)
        
        in_channels = model.head.classification_head.conv[0][0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        model.head.classification_head.num_classes = num_classes
        
        if dropout_prob is not None:
            class ConvReluDrop(nn.Sequential):
                def __init__(self, in_c, out_c, p):
                    super().__init__(
                        nn.Conv2d(in_c, out_c, 3, 1, 1),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p)
                    )
            conv_seq = list(model.head.classification_head.conv.children())
            last = conv_seq[-1][0]
            conv_seq[-1] = ConvReluDrop(last.in_channels, last.out_channels, float(dropout_prob))
            model.head.classification_head.conv = nn.Sequential(*conv_seq)

    return model


# -------- validation loss path (keep your manual forward similar to your helper) --------

@torch.no_grad()
def _eval_forward_retinanet(model, images, targets):
    """
    Manual forward in eval to compute losses.
    """
    from collections import OrderedDict

    model.eval()

    # transform
    images, targets = model.transform(images, targets)

    # backbone + FPN
    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    features = list(features.values())

    # heads
    cls_logits = model.head.classification_head(features)
    bbox_regression = model.head.regression_head(features)

    # anchors
    anchors = model.anchor_generator(images, features)

    # losses
    losses = model.compute_loss(targets,
                                {"cls_logits": cls_logits, "bbox_regression": bbox_regression},
                                anchors)
    return losses


@torch.no_grad()
def evaluate_loss_retinanet(model, data_loader, device):
    total = 0.0
    n = 0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = _eval_forward_retinanet(model, images, targets)
        loss = sum(loss_dict.values())
        total += float(loss)
        n += 1
    return total / max(n, 1)


def _eval_map50(model, data_loader, device):
    """Use tvrefs COCO evaluator; try to pull AP50 (index 1)."""
    out = tv_evaluate(model, data_loader, device=device)
    try:
        return float(out.coco_eval["bbox"].stats[1])
    except Exception:
        try:
            return float(out)
        except Exception:
            return 0.0


def save_experiment_specifications(run_dir: str, model_type: str, batch_size: int, lr: float, lf: float,
                                   dropout_prob: float, learning_schedule: str, dataset: str, epochs: int):
    specs = f"""Experiment Specifications:
    Model Type: {model_type}
    Batch Size: {batch_size}
    Learning Rate: {lr}
    Dropout rate: {dropout_prob}
    Lf: {lf}
    Learning Schedule: {learning_schedule}
    Dataset: {dataset}
    Epochs: {epochs}
"""
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(run_dir) / "experiment_specifications.txt", "w") as f:
        f.write(specs)


# ----------------------------------- Trainer -----------------------------------

def train_kfold(cfg: Dict):
    """
    cfg keys:
      - strategy: strict|smoothed|loose
      - aug_root: P:/BruiseDet_Repo/data/augmented
      - project_dir: P:/BruiseDet_Repo/outputs/training/{strategy}/RetinaNet
      - out_dir: alias of project_dir
      - resize_number: 640
      - batch: 8
      - workers: 16
      - threads: optional torch.set_num_threads
      - epochs: 30
      - lr: 0.001
      - lrf: 0.01
      - dropout_prob: 0.3 (None to disable)
      - pretrained: true
      - trainable_backbone_layers: 0
      - device: optional
      - init_from_prev: false, whether to initialize from previous weights
      - prev_weights_template: "", template path for previous weights
    """
    # device / threads
    device_override = cfg.get("device")
    device = torch.device(device_override) if device_override else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threads = int(cfg.get("threads", 0) or 0)
    if threads > 0:
        try:
            torch.set_num_threads(threads)
        except Exception:
            pass

    # paths
    strategy = str(cfg.get("strategy", "strict")).lower()
    strat_cap = strategy.capitalize()

    aug_root = Path(cfg.get("aug_root", r"P:/BruiseDet_Repo/data/augmented"))
    folds_root = aug_root / strat_cap

    proj_tpl = cfg.get("project_dir", r"P:/BruiseDet_Repo/outputs/training/{strategy}/RetinaNet")
    out_tpl = cfg.get("out_dir", proj_tpl)
    project_dir = Path(proj_tpl.format(strategy=strat_cap))
    out_dir = Path(out_tpl.format(strategy=strat_cap))
    project_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # hparams
    resize_number = int(cfg.get("resize_number", 640))
    batch = int(cfg.get("batch", 8))
    workers = int(cfg.get("workers", 16))
    epochs = int(cfg.get("epochs", 30))
    lr = float(cfg.get("lr", 0.001))
    lrf = float(cfg.get("lrf", 0.01))
    dropout_prob = cfg.get("dropout_prob", 0.3)
    pretrained = bool(cfg.get("pretrained", True))
    trainable_backbone_layers = int(cfg.get("trainable_backbone_layers", 0))
    
    # fine-tuning parameters
    init_from_prev = bool(cfg.get("init_from_prev", False))
    prev_weights_template = cfg.get("prev_weights_template", "")

    # discover folds
    if not folds_root.exists():
        raise FileNotFoundError(f"No augmented data for strategy '{strat_cap}' at: {folds_root}")
    kfold_dirs = sorted([p for p in folds_root.glob("Kfold_*") if p.is_dir()], key=lambda p: p.name)
    if not kfold_dirs:
        raise FileNotFoundError(f"No Kfold_* splits found under: {folds_root}")

    print(f"[RetinaNet] strategy={strat_cap}")
    print(f"[RetinaNet] data root: {folds_root}")
    print(f"[RetinaNet] outputs:   {project_dir}")
    if init_from_prev:
        print(f"[RetinaNet] initializing from previous weights: {prev_weights_template}")

    results = {}

    for k_idx, kdir in enumerate(kfold_dirs, start=1):
        train_dir = kdir / "train"
        val_dir = kdir / "val"
        if not (train_dir.exists() and val_dir.exists()):
            print(f"[warn] missing train/val under {kdir}, skipping")
            continue

        fold_dir = project_dir / f"k{k_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Determine weights path for this fold if init_from_prev is True
        weights_path = None
        if init_from_prev and prev_weights_template:
            try:
                weights_path = prev_weights_template.format(strategy=strat_cap, fold=k_idx)
                if not os.path.exists(weights_path):
                    print(f"[warn] Pre-trained weights not found at: {weights_path}, using default initialization")
                    weights_path = None
                else:
                    print(f"[RetinaNet|k{k_idx}] Loading pre-trained weights from: {weights_path}")
            except KeyError as e:
                print(f"[warn] Invalid placeholder in weights template: {e}, using default initialization")
                weights_path = None

        # datasets & loaders
        ds_train = IMPACT_AI(str(train_dir), resize_number=resize_number, cocoval=False)
        ds_val = IMPACT_AI(str(val_dir), resize_number=resize_number, cocoval=False)
        ds_val_coco = IMPACT_AI(str(val_dir), resize_number=resize_number, cocoval=True)

        dl_train = torch.utils.data.DataLoader(
            ds_train, batch_size=batch, shuffle=True, num_workers=workers,
            pin_memory=True, collate_fn=utils.collate_fn
        )
        dl_val = torch.utils.data.DataLoader(
            ds_val, batch_size=1, shuffle=False, num_workers=workers,
            pin_memory=True, collate_fn=utils.collate_fn
        )
        dl_val_coco = torch.utils.data.DataLoader(
            ds_val_coco, batch_size=1, shuffle=False, num_workers=workers,
            pin_memory=True, collate_fn=utils.collate_fn
        )

        # model
        model = initialize_retinanet(
            num_classes=2,
            pretrained=pretrained if weights_path is None else False,  # Don't use COCO weights if loading custom weights
            trainable_backbone_layers=trainable_backbone_layers,
            dropout_prob=dropout_prob,
            weights_path=weights_path
        ).to(device)

        # optim & sched
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
        lf = lambda x: max(1 - x / max(epochs, 1), 0) * (1.0 - lrf) + lrf
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        # files
        best_model_path = fold_dir / "best_model.pth"
        checkpoint_path = fold_dir / "checkpoint.pth"
        epoch_losses_path = fold_dir / "epoch_losses.csv"
        map_per_epoch_path = fold_dir / "mAP_per_epoch.csv"

        # resume (optional)
        start_epoch = 0
        best_map = 0.0
        if checkpoint_path.exists():
            print(f"[RetinaNet|k{k_idx}] resuming from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            best_map = float(ckpt.get("best_map", 0.0))

        # logs
        epoch_losses = pd.read_csv(epoch_losses_path).values.tolist() if epoch_losses_path.exists() else []
        map_hist = pd.read_csv(map_per_epoch_path).values.tolist() if map_per_epoch_path.exists() else []
        if map_hist:
            best_map = max(float(v) for _, v in map_hist)

        # save run specs
        save_experiment_specifications(
            str(fold_dir),
            model_type="retinanet_resnet50_fpn",
            batch_size=batch,
            lr=lr,
            lf=lrf,
            dropout_prob=0.0 if dropout_prob is None else float(dropout_prob),
            learning_schedule="LambdaLR",
            dataset=str(train_dir),
            epochs=epochs,
        )

        # train
        for epoch in range(start_epoch, epochs):
            torch.cuda.empty_cache()

            tv_train_one_epoch(
                model, optimizer, dl_train, device, epoch,
                run_dir=str(fold_dir), print_freq=10
            )

            val_loss = evaluate_loss_retinanet(model, dl_val, device=device)
            epoch_losses.append((epoch, float(val_loss)))

            map50 = _eval_map50(model, dl_val_coco, device=device)
            map_hist.append((epoch, float(map50)))

            if map50 > best_map:
                best_map = map50
                torch.save(model.state_dict(), best_model_path)
                print(f"[k{k_idx}] epoch {epoch}: new best mAP@0.5 {best_map:.4f} -> {best_model_path}")

            # checkpoint
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_map": best_map,
            }, checkpoint_path)

            lr_scheduler.step()
            print(f"[k{k_idx}] epoch {epoch+1}/{epochs} | lr={lr_scheduler.get_last_lr()[0]:.6f} | "
                  f"val_loss={float(val_loss):.4f} | mAP@0.5={map50:.4f}")

            # persist logs
            pd.DataFrame(epoch_losses, columns=["epoch", "Validation Loss"]).to_csv(epoch_losses_path, index=False)
            pd.DataFrame(map_hist, columns=["epoch", "mAP0.5"]).to_csv(map_per_epoch_path, index=False)

        results[f"k{k_idx}"] = {"best_map_50": best_map}

    # summary
    (project_dir / "summary.json").write_text(pd.io.json.dumps(results, indent=2))
    print(f"[RetinaNet] done. summary -> {project_dir / 'summary.json'}")
    return results