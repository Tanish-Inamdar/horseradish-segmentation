import os
import torch
import math
import json
import random
import trackio
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torchvision import datasets
from dataclasses import dataclass
from typing import List, Dict, Any
from torch_lr_finder import LRFinder
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoImageProcessor, AutoModel, AutoConfig, get_cosine_schedule_with_warmup
from model import DinoV3ForSegmentation
from segmentationDataset import HorseradishSegmentationDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

###TRAINING FOR LAB COMPUTER###
data_dir = "/home/tanishi2/ag group/dataset"
train_dir = "/home/tanishi2/ag group/dataset/train"
val_dir = "/home/tanishi2/ag group/dataset/val"
###TRAINING FOR LAB COMPUTER###

###TRAINING FOR PERSONAL###
# data_dir = "C:\\Users\\tanis\\AG GROUP\\horseradish_dataset"
# train_dir = "C:\\Users\\tanis\\AG GROUP\\horseradish_dataset\\train"
# val_dir = "C:\\Users\\tanis\\AG GROUP\\horseradish_dataset\\val"
###TRAINING FOR PERSONAL###



NUM_CLASSES = 3

# MODEL_NAME = "C:\\Users\\tanis\\AG GROUP\\dinov3-convnext-tiny-pretrain-lvd1689m"

MODEL_NAME = "facebook/dinov3-convnext-base-pretrain-lvd1689m"

# MODEL_NAME = "facebook/dinov3-convnext-large-pretrain-lvd1689m"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
backbone = AutoModel.from_pretrained(MODEL_NAME)
image_processor_config = json.loads(image_processor.to_json_string())
backbone_config = json.loads(AutoConfig.from_pretrained(MODEL_NAME).to_json_string())

train_dataset = HorseradishSegmentationDataset(root_dir=train_dir, processor=image_processor)
val_dataset = HorseradishSegmentationDataset(root_dir=val_dir, processor=image_processor)

freeze_backbone = True
model = DinoV3ForSegmentation(model_name=MODEL_NAME, num_classes=NUM_CLASSES)
model.to(device)
BATCH_SIZE = 8
NUM_WORKERS = 4
# NUM_WORKERS = 0
EPOCHS = 100000000
HEAD_LR = 5e-4
FULL_MODEL_LR = 1e-5
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.05
CHECKPOINT_DIR = "./weights"
EVAL_EVERY_STEPS = 100
UNFREEZE_AT_EPOCH = 20

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

@dataclass
class SegmentationCollator:
    def __call__(self, batch):
        # The dataset already returns a dictionary, so we just need to stack them
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}

collate_fn = SegmentationCollator()


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=collate_fn,
)

class CombinedLoss(nn.Module):
    def __init__(self, smooth=1e-6, num_classes=3, ce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        class_weights = torch.tensor([0.1, 0.5, 0.4]).to(device)
        self.smooth = smooth
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy(inputs, targets)
        inputs_softmax = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (inputs_softmax * targets_one_hot).sum()
        total_pixels = inputs_softmax.sum() + targets_one_hot.sum()
        dice = (2. * intersection + self.smooth) / (total_pixels + self.smooth)
        dice_loss = 1 - dice
        return (self.ce_weight * ce_loss) + (self.dice_weight * dice_loss)
    
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=4/3, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets_one_hot).sum()
        FP = ((1 - targets_one_hot) * inputs).sum()
        FN = (targets_one_hot * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        FocalTversky = (1 - Tversky)**self.gamma

        return FocalTversky
    
class CombinedLoss2(nn.Module):
    def __init__(self, smooth=1e-6, num_classes=3, ce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss2, self).__init__()
        class_weights = torch.tensor([0.05, 0.6, 0.35]).to(device)
        self.smooth = smooth
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights) 
    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy(inputs, targets)
        inputs_softmax = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dice_scores_per_class = []
        for cls in range(1, self.num_classes): 
            pred_c = inputs_softmax[:, cls, :, :]
            target_c = targets_one_hot[:, cls, :, :]
            intersection = (pred_c * target_c).sum(dim=[1, 2])
            union = pred_c.sum(dim=[1, 2]) + target_c.sum(dim=[1, 2])
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_scores_per_class.append(dice.mean()) 
        avg_dice_coefficient = torch.stack(dice_scores_per_class).mean()
        dice_loss = 1.0 - avg_dice_coefficient
        return (self.ce_weight * ce_loss) + (self.dice_weight * dice_loss)


###LOSS TESTING ###

# criterion = FocalTverskyLoss(alpha=0.7, beta=0.3).to(device)
criterion = CombinedLoss2(ce_weight=0.5, dice_weight=0.5, num_classes=NUM_CLASSES).to(device)
# criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5, num_classes=NUM_CLASSES).to(device)


scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

def dice_coefficient(pred, target, num_classes, smooth=1e-6):
    pred_softmax = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2)
    dice_scores = []
    for cls in range(1, num_classes):
        pred_c = pred_softmax[:, cls, :, :]
        target_c = target_one_hot[:, cls, :, :]
        if target_c.sum() > 0:
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice = (2. * intersection + smooth) / (union + smooth)
            dice_scores.append(dice.item())
    return np.mean(dice_scores) if dice_scores else 0.0

def evaluate() -> Dict[str, float]:
    model.eval()
    total_dice, loss_sum = 0.0, 0.0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * pixel_values.size(0)
            dice = dice_coefficient(logits, labels, num_classes=NUM_CLASSES)
            total_dice += dice
    avg_loss = loss_sum / len(val_dataset)
    avg_dice = total_dice / len(val_loader)
    return {"val_loss": avg_loss, "val_dice": avg_dice}


### TRAINING SCRIPT ###
if __name__ == '__main__':
    CKPT_PATH = os.path.join(CHECKPOINT_DIR, "model_best.pt")
    start_epoch = 1
    best_acc = 0.0
    optimizer = None
    scheduler = None
    if os.path.exists(CKPT_PATH):
        print("Checkpoint found. Loading model state...")
        checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        if start_epoch >= UNFREEZE_AT_EPOCH:
            print(f"Resuming after unfreeze epoch ({UNFREEZE_AT_EPOCH}). Unfreezing backbone and using full model LR.")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=FULL_MODEL_LR, weight_decay=WEIGHT_DECAY)
        else:
            print(f"Resuming before unfreeze epoch ({UNFREEZE_AT_EPOCH}). Keeping backbone frozen and using head LR.")
            model.freeze_backbone()
            optimizer = torch.optim.AdamW(model.decoder_head.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded optimizer state. Resuming training from epoch {start_epoch} | Best Dice so far: {best_acc*100:.2f}%")
        else:
            print(f"Resuming training from epoch {start_epoch} | Best Dice so far: {best_acc*100:.2f}% (Optimizer state not found).")
    else:
        # No checkpoint, start from scratch
        print("No checkpoint found. Starting training from scratch.")
        model.freeze_backbone()
        optimizer = torch.optim.AdamW(model.decoder_head.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)    
    
    if start_epoch >= UNFREEZE_AT_EPOCH:
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=40
        )
        if os.path.exists(CKPT_PATH) and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scheduler.best = best_acc
    
    trackio.init(project="horseradish-dinov3", config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "head_lr": HEAD_LR,
        "full_model_lr": FULL_MODEL_LR,
        "unfreeze_at_epoch": UNFREEZE_AT_EPOCH
    })

    for epoch in range(start_epoch, EPOCHS + 1):
        if epoch == UNFREEZE_AT_EPOCH:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=FULL_MODEL_LR, weight_decay=WEIGHT_DECAY)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=40)

        model.train()
        if epoch < UNFREEZE_AT_EPOCH:
            model.backbone.eval()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                logits = model(pixel_values)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        metrics = evaluate()
        avg_train_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"\n[END OF EPOCH {epoch}] Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {metrics['val_loss']:.4f} | Val Dice: {metrics['val_dice']*100:.2f}% | LR: {current_lr:.2e}"
        )

        trackio.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": metrics['val_loss'],
            "val_dice": metrics['val_dice'],
            "learning_rate": current_lr
        })
        if scheduler is not None:
            scheduler.step(metrics['val_dice'])
        if metrics["val_dice"] > best_acc:
            print(f"New best model. Dice improved from {best_acc*100:.2f}% to {metrics['val_dice']*100:.2f}%. Saving checkpoint...")
            best_acc = metrics["val_dice"]
            save_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
                "config": {"model_name": MODEL_NAME}
            }
            if scheduler is not None:
                save_dict['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(save_dict, CKPT_PATH)


    trackio.finish()


