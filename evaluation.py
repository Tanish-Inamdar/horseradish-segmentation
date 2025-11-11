import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
import matplotlib.image as mpimg
from matplotlib.image import AxesImage
from model import DinoV3ForSegmentation
from segmentationDataset import HorseradishSegmentationDataset
from training import SegmentationCollator
import cv2

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
WEED_CLASS_ID = 2


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

def extract_weed_centroids(binary_weed_mask, min_area_threshold=50):
    #need 8bit mask for cv2
    mask_8bit = (binary_weed_mask > 0).astype(np.uint8) * 255 
    contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area_threshold:
            # Calculate moments for the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                H, W = binary_weed_mask.shape
                norm_cX = cX / W
                norm_cY = cY / H
                
                centroids.append((norm_cX, norm_cY, area))
                
    return centroids

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
            
            vis_logits, vis_features = model(pixel_values)
            vis_preds = torch.argmax(vis_logits, dim=1).cpu().numpy()
            
            dice, iou = calculate_metrics(vis_logits.cpu(), labels.cpu(), num_classes=NUM_CLASSES)
            total_dice += dice
            total_iou += iou

    avg_dice = total_dice / len(val_loader)
    avg_iou = total_iou / len(val_loader)

    print(f"\n--- Evaluation Complete ---")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average Mean IoU: {avg_iou:.4f}")

    print("\nVisualizing some random predictions...")
    num_examples = min(BATCH_SIZE, 2)

    vis_batch = next(iter(val_loader))

    ground_truth_mask = vis_batch["labels"][0].numpy()
    gt_pil = Image.fromarray((ground_truth_mask * 120).astype(np.uint8))
    # gt_pil.save("ground_truth_sanity_check.jpg")
    # print("Saved a ground truth mask to ground_truth_sanity_check.jpg")
    pixel_values = vis_batch["pixel_values"][:num_examples].to(DEVICE)
    
    with torch.no_grad():
        vis_logits, vis_features = model(pixel_values)
        vis_preds = torch.argmax(vis_logits, dim=1).cpu().numpy()

    mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
    std = torch.tensor(image_processor.image_std).view(3, 1, 1)
    vis_images = (pixel_values.cpu() * std) + mean
    vis_images = (vis_images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

    fig, axes = plt.subplots(num_examples, 3, figsize=(6, num_examples * 2))
    for i in range(num_examples):
        img_pil = Image.fromarray(vis_images[i])
        pred_mask = vis_preds[i]

        weed_mask_255 = (pred_mask == WEED_CLASS_ID).astype(np.uint8) * 255
        
        # 2. Extract the centroids
        weed_centroids = extract_weed_centroids(weed_mask_255, min_area_threshold=100) #maybe change min area on testing?
        
        ###test###
        print(f"Image {i+1}: Found {len(weed_centroids)} weed centroids (normalized x, y, area):")
        print(weed_centroids)
        ###test###

        pred_mask_resized = Image.fromarray(pred_mask.astype(np.uint8)).resize(
            img_pil.size, Image.NEAREST
        )
        
        overlay = create_overlay(img_pil, np.array(pred_mask_resized), COLOR_MAP)
        
        #logic for feature map?#
        features = vis_features[i].cpu()
        H_feat, W_feat = features.shape[-2], features.shape[-1]
        pred_mask_downsampled = Image.fromarray(vis_preds[i].astype(np.uint8)).resize(
            (W_feat, H_feat), Image.NEAREST
        )
        pred_mask_downsampled = np.array(pred_mask_downsampled)
        plant_coords = np.argwhere(pred_mask_downsampled == 1)

        if len(plant_coords) > 0:
            ref_h, ref_w = plant_coords[len(plant_coords) // 2]
        else:
            ref_h, ref_w = H_feat // 2, W_feat // 2
        
        ref_feature = features[:, ref_h, ref_w]
        flat_features = features.permute(1, 2, 0).reshape(-1, features.shape[0])
        ref_feature_norm = F.normalize(ref_feature.unsqueeze(0), p=2, dim=1)
        flat_features_norm = F.normalize(flat_features, p=2, dim=1)
        similarity_map = torch.matmul(flat_features_norm, ref_feature_norm.T).reshape(H_feat, W_feat).numpy()
        
        im = None
        
        ax = axes[i, 0]
        ax.imshow(img_pil)
        ax.set_title("Input Image")
        ax.axis('off')
        
        
        # Column 2: Predicted Segmentation Overlay 
        ax = axes[i, 1]
        ax.imshow(overlay)
        ax.set_title("Predicted Overlay")
        ax.axis('off')


        for norm_cX, norm_cY, area in weed_centroids:
            # pixel_cX = norm_cX * img_width
            # pixel_cY = norm_cY * img_height
            ax.scatter(norm_cX, norm_cY, color='yellow', marker='o', s=50, linewidth=1.5) 
            #label text no idea?
            ax.text(norm_cX + 10, norm_cY + 10, f"{int(area)}", color='yellow', fontsize=6)

        # Column 3: Feature Similarity Map
        ax = axes[i, 2] 
        ax.imshow(img_pil) 
        im_current = ax.imshow(
            similarity_map, 
            alpha=0.6, 
            cmap='inferno',
            interpolation='bicubic',
            vmin=np.min(similarity_map), 
            vmax=1.0, # Cosine similarity ranges up to 1.0
            extent=[0, img_pil.size[0], img_pil.size[1], 0] # Map to original pixel space
        )
        ax.scatter(
            img_pil.size[0] * (ref_w + 0.5) / W_feat, 
            img_pil.size[1] * (ref_h + 0.5) / H_feat, 
            color='red', 
            marker='x', 
            s=100, 
            linewidth=2
        )
        if i == num_examples - 1:
            im = im_current
        
        ax.set_title(f"Feature Sim. (Ref: {ref_h},{ref_w})")
        ax.axis('off')

    plt.tight_layout()
    if im is not None:
            cbar = fig.colorbar(im, ax=axes[:, 2].ravel().tolist(), orientation='vertical', 
                                fraction=0.046, pad=0.04)
            cbar.set_label('Cosine Similarity (Dense Features)')
    output_filename = "horseradish_analysis.png"
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    print(f"\nSaved high-resolution visualization to {output_filename}")
    plt.show()