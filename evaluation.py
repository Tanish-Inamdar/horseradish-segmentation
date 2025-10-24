import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from model import DinoV3ForSegmentation
from segmentationDataset import HorseradishSegmentationDataset
from training import SegmentationCollator

# VAL_DIR = "C:\\Users\\tanis\\AG GROUP\\horseradish_dataset\\val"
VAL_DIR = "/home/tanishi2/ag group/dataset/val"
CKPT_PATH = "./weights/model_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_CLASSES = 3 

COLOR_MAP = {
    0: (0, 0, 0),        # Black for Background
    1: (0, 255, 0),      # Green for Horseradish
    2: (255, 0, 0),      # Red for Weed
}


print("Loading model and dataset...")
checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
image_processor = AutoImageProcessor.from_pretrained(
    checkpoint["config"]["model_name"]
)

val_dataset = HorseradishSegmentationDataset(root_dir=VAL_DIR, processor=image_processor)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4, # set to 0 now change later
    # num_workers=0,
    collate_fn=SegmentationCollator(),
)

model = DinoV3ForSegmentation(
    model_name=checkpoint["config"]["model_name"],
    num_classes=NUM_CLASSES
).to(DEVICE)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model and data loaded successfully!")


def calculate_metrics(pred_logits, target_masks, num_classes, smooth=1e-6):
    """Calculates both Dice and Mean IoU for a batch."""
    pred_masks = torch.argmax(pred_logits, dim=1)
   
    target_one_hot = F.one_hot(target_masks, num_classes=num_classes).permute(0, 3, 1, 2)
    pred_one_hot = F.one_hot(pred_masks, num_classes=num_classes).permute(0, 3, 1, 2)

    dice_scores = []
    iou_scores = []
    for cls in range(1, num_classes): 
        pred_c = pred_one_hot[:, cls, :, :]
        target_c = target_one_hot[:, cls, :, :]
        if target_c.sum() > 0:
            intersection = (pred_c * target_c).sum()
            total_pixels = pred_c.sum() + target_c.sum()

            dice = (2. * intersection + smooth) / (total_pixels + smooth)
            dice_scores.append(dice.item())

            union = total_pixels - intersection
            iou = (intersection + smooth) / (union + smooth)
            iou_scores.append(iou.item())
    
    mean_dice = np.mean(dice_scores) if dice_scores else 0.0
    mean_iou = np.mean(iou_scores) if iou_scores else 0.0
    return mean_dice, mean_iou

def create_overlay(image, mask, color_map, alpha=0.5):
    """Creates a visual overlay of the mask on the image."""
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx, color in color_map.items():
        rgb_mask[mask == class_idx] = color
    
    mask_pil = Image.fromarray(rgb_mask)
    overlay = Image.blend(image.convert("RGB"), mask_pil, alpha=alpha)
    return overlay

if __name__ == "__main__":
    total_dice = 0.0
    total_iou = 0.0
    
    print("\nStarting evaluation...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            logits = model(pixel_values)
            
            dice, iou = calculate_metrics(logits.cpu(), labels.cpu(), num_classes=NUM_CLASSES)
            total_dice += dice
            total_iou += iou

    avg_dice = total_dice / len(val_loader)
    avg_iou = total_iou / len(val_loader)

    print(f"\n--- Evaluation Complete ---")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average Mean IoU: {avg_iou:.4f}")

    print("\nVisualizing some random predictions...")
    num_examples = min(BATCH_SIZE, 4) 

    vis_batch = next(iter(val_loader))
    ground_truth_mask = vis_batch["labels"][0].numpy()
    gt_pil = Image.fromarray((ground_truth_mask * 120).astype(np.uint8))
    gt_pil.save("ground_truth_sanity_check.jpg")
    print("Saved a ground truth mask to ground_truth_sanity_check.jpg")
    pixel_values = vis_batch["pixel_values"][:num_examples].to(DEVICE)
    
    with torch.no_grad():
        vis_logits = model(pixel_values)
        vis_preds = torch.argmax(vis_logits, dim=1).cpu().numpy()

    mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
    std = torch.tensor(image_processor.image_std).view(3, 1, 1)
    vis_images = (pixel_values.cpu() * std) + mean
    vis_images = (vis_images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

    fig, axes = plt.subplots(num_examples, 2, figsize=(8, num_examples * 2))
    for i in range(num_examples):
        img_pil = Image.fromarray(vis_images[i])
        pred_mask = vis_preds[i]
        pred_mask_resized = Image.fromarray(pred_mask.astype(np.uint8)).resize(
            img_pil.size, Image.NEAREST
        )
        
        overlay = create_overlay(img_pil, np.array(pred_mask_resized), COLOR_MAP)
        
        ax = axes[i, 0]
        ax.imshow(img_pil)
        ax.set_title("Input Image")
        ax.axis('off')
        
        ax = axes[i, 1]
        ax.imshow(overlay)
        ax.set_title("Predicted Overlay")
        ax.axis('off')

    plt.tight_layout()
    plt.show()