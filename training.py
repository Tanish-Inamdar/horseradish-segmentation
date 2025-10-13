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
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoImageProcessor, AutoModel, AutoConfig, get_cosine_schedule_with_warmup
from model import DinoV3ForSegmentation
from segmentationDataset import HorseradishSegmentationDataset


data_dir = "/home/tanishi2/ag group/dataset"
train_dir = "/home/tanishi2/ag group/dataset/train"
val_dir = "/home/tanishi2/ag group/dataset/val"




NUM_CLASSES = 3

# MODEL_NAME = "C:\\Users\\tanis\\AG GROUP\\dinov3-convnext-tiny-pretrain-lvd1689m"
MODEL_NAME = "facebook/dinov3-convnext-base-pretrain-lvd1689m"
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
BATCH_SIZE = 32
# NUM_WORKERS = min(8, os.cpu_count() or 2)
NUM_WORKERS = 4
EPOCHS = 50
LR = 5e-5
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.05
CHECKPOINT_DIR = "./weights"
EVAL_EVERY_STEPS = 100

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

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, num_classes=3):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        inputs_softmax = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        # Permute to match the input shape: (Batch, Num_Classes, H, W)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)
        intersection = (inputs_softmax * targets_one_hot).sum()
        total_pixels = inputs_softmax.sum() + targets_one_hot.sum()
        dice = (2. * intersection + self.smooth) / (total_pixels + self.smooth)
        loss = 1 - dice
        
        return loss

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = EPOCHS * math.ceil(len(train_loader))
warmup_steps = int(WARMUP_RATIO * total_steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
criterion = DiceLoss(num_classes=NUM_CLASSES)

scaler = torch.amp.GradScaler('cuda',enabled=torch.cuda.is_available())



import torch.nn.functional as F

def dice_coefficient(pred, target, num_classes, smooth=1e-6):
    
    pred_softmax = F.softmax(pred, dim=1)
    

    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2)

    # 3. Calculate Dice for each class (ignoring background class 0)
    dice_scores = []
    for cls in range(1, num_classes): # Start from 1 to ignore background
        pred_c = pred_softmax[:, cls, :, :]
        target_c = target_one_hot[:, cls, :, :]
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)
        
    # Average the Dice score across the foreground classes (horseradish and weed)
    return torch.mean(torch.tensor(dice_scores)).item()

def evaluate() -> Dict[str, float]:
    model.eval()
    total, loss_sum = 0, 0.0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)
            dice = dice_coefficient(logits, labels, num_classes = NUM_CLASSES)
            total += dice
            avg_loss = loss_sum / len(val_loader)
            avg_dice = total/ len(val_loader)
            
    return {
        "val_loss": avg_loss,
        "val_dice": avg_dice,
    }

CKPT_PATH = os.path.join(CHECKPOINT_DIR, "model_best.pt")
start_epoch = 1
global_step = 0
best_acc = 0.0

if __name__ == '__main__':
    trackio.init(project="dinov3", config={
        "epochs": EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE
    })
    # num_total_epochs = 50

    if os.path.exists(CKPT_PATH):
        print("Checkpoint found! Loading model state...")
        checkpoint = torch.load(CKPT_PATH, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['step']
        best_acc = checkpoint.get('best_acc', 0.0) 
        
        print(f"Resuming training from epoch {start_epoch} | Best Dice so far: {best_acc*100:.2f}%")
    else:
        print("No checkpoint found. Starting training from scratch.")



    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        model.backbone.eval()

        running_loss = 0.0
        for i, batch in tqdm(enumerate(train_loader, start=1), desc=f"Epoch {epoch}/{EPOCHS}", total=len(train_loader)):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values)
            loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % EVAL_EVERY_STEPS == 0:
                metrics = evaluate()
                print(
                    f"\n[epoch {epoch} | step {global_step}] "
                    f"train_loss={running_loss / EVAL_EVERY_STEPS:.4f} "
                    f"val_loss={metrics['val_loss']:.4f} val_dice={metrics['val_dice']*100:.2f}%"
                )
                
                trackio.log({
                    "epoch": epoch,
                    "val_dice": metrics['val_dice'],
                    "train_loss": running_loss / EVAL_EVERY_STEPS,
                    "val_loss": metrics['val_loss'],
                    "learning_rate": scheduler.get_last_lr()[0]
                })

                running_loss = 0.0

                if metrics["val_dice"] > best_acc:
                    print(f" New best model found! Dice improved from {best_acc*100:.2f}% to {metrics['val_dice']*100:.2f}%. Saving checkpoint...")
                    best_acc = metrics["val_dice"]
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "config": {
                            "model_name": MODEL_NAME,
                            "backbone": backbone_config,
                            "image_processor": image_processor_config,
                            "freeze_backbone": freeze_backbone,
                        },
                        "step": global_step,
                        "epoch": epoch,
                        "best_acc": best_acc,
                    }, CKPT_PATH)

        metrics = evaluate()
        print(
            f"END EPOCH {epoch}: val_loss={metrics['val_loss']:.4f} val_dice={metrics['val_dice']*100:.2f}% "
            f"(best_acc={best_acc*100:.2f}%)"
        )
    
    trackio.finish()
