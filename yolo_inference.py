import torch
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, Colors
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = "/home/tanishi2/ag group/horseradish-segmentation/YOLO_Distilled_Training/yolov8m_from_dinov3_teacher200epochs/weights/best.pt"
IMAGE_TO_TEST = "/home/tanishi2/Downloads/DJI_20250612151222_0175_D.JPG"
OUTPUT_SAVE_PATH = "yolo_segmented_prod_output200epoch.jpg"

WEED_CLASS_ID = 1  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- HELPER FUNCTION: IOU CALCULATION ---
def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) between two boxes.
    Box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0: return 0
    return intersection_area / union_area

# --- MAIN PIPELINE ---
print(f"Loading model {MODEL_PATH} to {DEVICE}...")
model = YOLO(MODEL_PATH).to(DEVICE)

original_image_pil = Image.open(IMAGE_TO_TEST).convert("RGB")
original_image_np = np.array(original_image_pil)

print(f"Running inference on GPU...")
# KEY CHANGE 1: Added 'iou=0.5'. 
# This tells NMS: "If boxes overlap by 50% or more, suppress the lower confidence one."
# Combined with agnostic_nms=True, this is your first line of defense.
results = model.predict(
    original_image_np, 
    device=DEVICE, 
    verbose=False, 
    conf=0.15, 
    iou=0.5,            # <--- Lowered from default 0.7 to catch duplicates
    agnostic_nms=True   # <--- Treats Weed/Crop as same class for suppression
)

result = results[0].cpu() # Work with the single image result
print(f"Inference complete. Initial detection count: {len(result.boxes)}")

# --- KEY CHANGE 2: MANUAL CONFLICT FILTERING ---
# Sometimes NMS misses cases where a small box is inside a big box.
# We will manually filter the results before plotting.

keep_indices = []
if len(result.boxes) > 0:
    boxes_data = result.boxes.xyxy.numpy()
    conf_data = result.boxes.conf.numpy()
    cls_data = result.boxes.cls.numpy()
    
    # Start by assuming we keep everything
    should_keep = [True] * len(boxes_data)
    
    for i in range(len(boxes_data)):
        for j in range(i + 1, len(boxes_data)):
            if not should_keep[i] or not should_keep[j]:
                continue

            # Check IoU between Box i and Box j
            iou = calculate_iou(boxes_data[i], boxes_data[j])
            
            # If they overlap significantly (e.g., > 40%), kill the lower confidence one
            # This handles the "Weed detected on top of Horseradish" case
            if iou > 0.40: 
                if conf_data[i] > conf_data[j]:
                    should_keep[j] = False
                else:
                    should_keep[i] = False

    # Create a list of indices we actually want to keep
    keep_indices = [i for i, keep in enumerate(should_keep) if keep]

print(f"Post-filtering object count: {len(keep_indices)}")


# --- PLOTTING ---
annotator = Annotator(original_image_np.copy(), line_width=2)
colors = Colors()  

# Only proceed if we have objects left to plot
if len(keep_indices) > 0 and result.masks is not None:
    print("Plotting filtered masks...")
    
    target_height, target_width = annotator.im.shape[:2]
    
    # Filter the masks using our keep_indices
    # result.masks.data is (N, H, W). We select only the 'keep' indices.
    all_masks = result.masks.data.numpy()
    filtered_masks = all_masks[keep_indices]
    
    resized_masks = []
    for mask in filtered_masks:
        resized_mask = cv2.resize(
            mask, 
            (target_width, target_height), 
            interpolation=cv2.INTER_NEAREST 
        )
        resized_masks.append(resized_mask)
    
    if len(resized_masks) > 0:
        resized_masks_np = np.stack(resized_masks, axis=0)
        
        # Get class IDs for the kept objects
        all_classes = result.boxes.cls.int().tolist()
        filtered_classes = [all_classes[i] for i in keep_indices]
        
        mask_colors = [colors(idx, True) for idx in filtered_classes]

        annotator.masks(
            resized_masks_np, 
            colors=mask_colors,
            alpha=0.5
        )

# Plot Boxes and Centroids using the filtered indices
if len(keep_indices) > 0:
    for original_index in keep_indices:
        box = result.boxes[original_index]
        class_id = int(box.cls)
        label = f"{model.names[class_id]} {box.conf.item():.2f}"
        color = colors(class_id, True)
        
        annotator.box_label(box.xyxy.squeeze(), label, color=color)

        # Centroid logic (Preserved from your code)
        if class_id == WEED_CLASS_ID and result.masks is not None:
            # Note: masks.xy corresponds to the original index list
            mask_points = result.masks.xy[original_index]
            if len(mask_points) > 0:
                M = cv2.moments(mask_points.astype(np.int32))
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(annotator.im, (cX, cY), 5, (255, 255, 0), -1)

overlay_image = annotator.result()
overlay_image_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
cv2.imwrite(OUTPUT_SAVE_PATH, overlay_image_bgr)
print(f"Successfully saved production overlay image to {OUTPUT_SAVE_PATH}")