import torch
import numpy as np
from PIL import Image
import cv2
import os
from tqdm import tqdm
from model import DinoV3ForSegmentation
from transformers import AutoImageProcessor

CKPT_PATH = "/home/tanishi2/ag group/horseradish-segmentation/weights/dinov3_model_trained.pt"
MODEL_NAME = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = "/home/tanishi2/ag group/dataset/train/images"
OUTPUT_LABEL_DIR = "/home/tanishi2/ag group/dataset/train/pseudo_labels" 

WEED_CLASS_ID = 2
HORSERADISH_CLASS_ID = 1
CLASSES_TO_EXTRACT = [HORSERADISH_CLASS_ID, WEED_CLASS_ID] 
NUM_CLASSES = 3
MIN_AREA_THRESHOLD = 50 

def load_teacher_model(ckpt_path, model_name, num_classes, device):
    print("Loading DINOv3 teacher model from checkpoint...")
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = DinoV3ForSegmentation(
        model_name=model_name,
        num_classes=num_classes  
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Teacher model loaded successfully!")
    return model, image_processor

def infer_segmentation(image: Image.Image, model, processor, device):
    """Runs inference and returns the predicted segmentation mask."""
    original_size = image.size 
    
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(inputs["pixel_values"])
        logits = outputs[0]
        
        #resize logits to original size for better quality
        logits_upsampled = torch.nn.functional.interpolate(
            logits, 
            size=original_size[::-1], # (height, width)
            mode='bilinear', 
            align_corners=False
        )
        pred_mask = torch.argmax(logits_upsampled, dim=1)
        pred_mask = pred_mask.cpu().numpy().squeeze() 
    
    return pred_mask

def mask_to_yolo_format(mask: np.ndarray, width: int, height: int):
    """
    Converts a multi-class segmentation mask into a list of YOLO-formatted strings.
    """
    yolo_lines = []
    
    for class_id in CLASSES_TO_EXTRACT:
        binary_mask = (mask == class_id).astype(np.uint8) * 255
        #logic for centroids
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < MIN_AREA_THRESHOLD:
                continue
            
            points = contour.squeeze(axis=1)
            points_normalized = points.astype(np.float32)
            points_normalized[:, 0] /= width
            points_normalized[:, 1] /= height
            points_flat = points_normalized.ravel()

            #string formating
            line_points = " ".join([f"{p:.6f}" for p in points_flat])
            if class_id == HORSERADISH_CLASS_ID: # 1
                yolo_class_id = 0
            elif class_id == WEED_CLASS_ID: # 2
                yolo_class_id = 1
            else:
                continue
            yolo_line = f"{yolo_class_id} {line_points}"
            yolo_lines.append(yolo_line)
            
    return yolo_lines

if __name__ == "__main__":
    model, image_processor = load_teacher_model(CKPT_PATH, MODEL_NAME, NUM_CLASSES, DEVICE)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
    print(f"Saving pseudo-labels to: {OUTPUT_LABEL_DIR}")
    
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(image_files)} images to process...")

    for image_name in tqdm(image_files, desc="Generating Pseudo-Labels"):
        image_path = os.path.join(IMAGE_DIR, image_name)
        
        try:
            image = Image.open(image_path).convert("RGB")
            
            pred_mask = infer_segmentation(image, model, image_processor, DEVICE)
            
            yolo_lines = mask_to_yolo_format(pred_mask, image.width, image.height)
            
            if yolo_lines:
                base_name = os.path.splitext(image_name)[0]
                output_path = os.path.join(OUTPUT_LABEL_DIR, f"{base_name}.txt")
                
                with open(output_path, 'w') as f:
                    f.write("\n".join(yolo_lines))
                    
        except Exception as e:
            print(f"Failed to process {image_name}: {e}")
            
    print("\n--- Pseudo-label generation complete! ---")
    print(f"New labels are in: {OUTPUT_LABEL_DIR}")