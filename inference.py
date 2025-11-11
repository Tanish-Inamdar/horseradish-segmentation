import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import DinoV3ForSegmentation
from transformers import AutoImageProcessor
import cv2


CKPT_PATH = "./weights/model_best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_NAME = "facebook/dinov3-convnext-base-pretrain-lvd1689m"
MODEL_NAME = "facebook/dinov3-convnext-large-pretrain-lvd1689m"


# testing a picture from the validation set
# IMAGE_TO_TEST = "C:\\Users\\tanis\\AG GROUP\\horseradish_dataset\\val\\images\\DJI_20250612151152_0167_D_JPG.rf.8fb3f5876e4879eb7fc14a9775995071.jpg"
# IMAGE_TO_TEST = "/home/tanishi2/ag group/dataset/val/images/DJI_20250612150326_0055_D_JPG.rf.71aca717600db015bc238f39667d4210.jpg"
IMAGE_TO_TEST = "/home/tanishi2/Downloads/DJI_20250612150736_0107_D.JPG"
COLOR_MAP = {
    0: (0, 0, 0),        # Black for Background
    1: (0, 255, 0),      # Green for Horseradish
    2: (255, 0, 0),      # Red for Weed
}

WEED_CLASS_ID = 2



print("loading model from checkpoint...")
checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)

image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# image_processor = AutoImageProcessor.from_pretrained(
#     checkpoint["config"]["model_name"]
# )

model = DinoV3ForSegmentation(
    # model_name=checkpoint["config"]["model_name"],
    model_name=MODEL_NAME,
    num_classes=3  
).to(device)

# Load the trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("loaded modelsuccessfully!")

def infer_segmentation(image: Image.Image, model, processor, device):
    """Runs inference and returns the predicted segmentation mask."""
    original_size = image.size # (width, height)
    
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Get model predictions (logits)
        outputs = model(inputs["pixel_values"])
        logits = outputs[0]
        pred_mask = torch.argmax(logits, dim=1)
        pred_mask = pred_mask.cpu().numpy().squeeze() 
    mask_pil = Image.fromarray(pred_mask.astype(np.uint8)).resize(original_size, Image.NEAREST)
    
    return np.array(mask_pil)

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


def visualize_mask(original_image: Image.Image, mask: np.ndarray, color_map: dict):
    """Overlays the segmentation mask on the original image."""
    # Create an RGB image from the mask using the color map
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # weeds = np.where(2)
    # centroid = np.mean(weeds, axis = 0)


    for class_idx, color in color_map.items():
        rgb_mask[mask == class_idx] = color
    
    mask_pil = Image.fromarray(rgb_mask)
    # mask_cent = Image.fromarray(centroid)
    # mask_mix = Image.blend(mask_pil, mask_cent, alpha = 0.5)
    overlay = Image.blend(original_image.convert("RGB"), mask_pil, alpha=0.5)
    return overlay

if __name__ == "__main__":
    original_image = Image.open(IMAGE_TO_TEST).convert("RGB")

    print("Running inference...")
    pred_mask = infer_segmentation(original_image, model, image_processor, device)
    overlay_image = visualize_mask(original_image, pred_mask, COLOR_MAP)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    weed_mask_255 = (pred_mask == WEED_CLASS_ID).astype(np.uint8) * 255
    weed_centroids = extract_weed_centroids(weed_mask_255, min_area_threshold=50)

    ###test###
    print(f"Found {len(weed_centroids)} weed centroids (normalized x, y, area):")
    print(weed_centroids)
    ###test###


    
    ax[1].imshow(overlay_image)
    ax[1].set_title("Segmentation Overlay")
    ax[1].axis('off')

    img_width, img_height = original_image.size

    for norm_cX, norm_cY, area in weed_centroids:
        pixel_cX = norm_cX * img_width
        pixel_cY = norm_cY * img_height
        ax[1].scatter(pixel_cX, pixel_cY, color='yellow', marker='o', s=50, linewidth=1.5) 
        #label text no idea?
        ax[1].text(pixel_cX + 10, pixel_cY + 10, f"{int(area)}", color='yellow', fontsize=6)
    
    plt.tight_layout()
    plt.show()
    output_path = "segmented_output.jpg"
    overlay_image.save(output_path)
    print(f"Saved overlay image to {output_path}")