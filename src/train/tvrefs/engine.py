import math
import sys
import time
import pandas as pd
import torch
import torchvision.models.detection.mask_rcnn
from . import utils
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset
import os
from torchvision.ops import generalized_box_iou_loss

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, run_dir, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    losses_data = []  # Initialize the list to store loss data
    iteration = 0
    use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7  # Enable FP16 AMP
    scaler = torch.cuda.amp.GradScaler() if use_amp else None  # FP16 AMP uses GradScaler
    # Warmup scheduler (if necessary)
    warmup_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        iteration += 1

        # Reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        losses_data.append({
            "epoch": epoch,
            "iteration": iteration,  # Assuming this is tracked or manually implemented
            "total_loss": losses_reduced.item(),
            **{k: v.item() for k, v in loss_dict_reduced.items()}
        })

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if warmup_scheduler is not None:
            warmup_scheduler.step()
        #else:
           #optimizer.step()  # Step the main scheduler after each batch update

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    if losses_data:  # Ensure there is data to process
        df_losses_epoch = pd.DataFrame(losses_data)
        avg_losses = df_losses_epoch.mean().to_dict()
        avg_losses['epoch'] = epoch  # Add epoch number to the average losses

        # Save averages to CSV
        filename = os.path.join(run_dir, "average_training_losses.csv")
        if os.path.exists(filename):
            df_existing = pd.read_csv(filename)
            df_new_row = pd.DataFrame([avg_losses])
            df_losses = pd.concat([df_existing, df_new_row], ignore_index=True)
        else:
            df_losses = pd.DataFrame([avg_losses])
        
        df_losses.to_csv(filename, index=False)

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()
    
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    
    # Extract mAP0.5 and return it directly
    mAP0_5 = coco_evaluator.coco_eval[iou_types[0]].stats[1]  # Assuming iou_type is 'bbox'
    
    torch.set_num_threads(n_threads)

    return mAP0_5

@torch.inference_mode()
def evaluate_other(model, data_loader, device):
    import numpy as np

    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # Get COCO dataset
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    print("COCO Dataset Image IDs (First 10):", coco.getImgIds()[:10])

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        # Extract image IDs
        image_ids = [int(t["image_id"].item()) for t in targets]
        print("Predicted Image IDs:", image_ids)

        # Convert outputs to COCO format
        formatted_results = {}
        for img_id, output in zip(image_ids, outputs):
            formatted_results[img_id] = []
            for i in range(len(output["boxes"])):
                formatted_results[img_id].append({
                    "image_id": img_id,
                    "category_id": int(output["labels"][i].item()),
                    "bbox": output["boxes"][i].tolist(),
                    "score": float(output["scores"][i].item())
                })

        print("Formatted Results (First 5 for each image):")
        for k, v in list(formatted_results.items())[:5]:  # Print first 5 images' results
            print(f"Image {k}: {v[:5]}")

        if not formatted_results:
            print("⚠️ WARNING: No detections found. Skipping batch.")
            continue  # Skip this batch

        evaluator_time = time.time()
        coco_evaluator.update(sum(formatted_results.values(), []))
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()
    
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    
    # Extract mAP0.5
    mAP0_5 = coco_evaluator.coco_eval[iou_types[0]].stats[1]
    
    torch.set_num_threads(n_threads)

    return mAP0_5


