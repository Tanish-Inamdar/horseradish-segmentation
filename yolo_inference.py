import torch
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, Colors
from PIL import Image


MODEL_PATH = "YOLO_Distilled_Training/yolov8n_from_dinov3_teacher/weights/best.pt"
IMAGE_TO_TEST = "/home/tanishi2/Downloads/DJI_20250612151222_0175_D.JPG"
OUTPUT_SAVE_PATH = "yolo_segmented_prod_output.jpg"

WEED_CLASS_ID = 1  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(f"Loading model {MODEL_PATH} to {DEVICE}...")
model = YOLO(MODEL_PATH).to(DEVICE)

original_image_pil = Image.open(IMAGE_TO_TEST).convert("RGB")
original_image_np = np.array(original_image_pil)

print(f"Running inference on GPU...")
results = model.predict(original_image_np, device=DEVICE, verbose=False)

results = results[0].cpu()
print(f"Inference complete. Plotting {len(results.boxes)} objects on CPU...")

annotator = Annotator(original_image_np.copy(), line_width=2)
colors = Colors()  

if results.masks:
    print("Plotting masks...")
    
    # Get the target shape (H, W) from the annotator's original image
    target_height, target_width = annotator.im.shape[:2]
    
    # Manually resize all masks to match the annotator's image size
    resized_masks = []
    # results.masks.data is (N, 480, 640)
    for mask in results.masks.data.numpy():
        # mask shape is (480, 640)
        # target shape for cv2.resize is (W, H)
        resized_mask = cv2.resize(
            mask, 
            (target_width, target_height), 
            interpolation=cv2.INTER_NEAREST # Use NEAREST to keep binary 0/1 values
        )
        resized_masks.append(resized_mask)
    
    # Stack them back into a single (N, H, W) numpy array
    resized_masks_np = np.stack(resized_masks, axis=0)
    
    # Get the class IDs for all masks
    class_indices = results.boxes.cls.int().tolist()
    
    # Create the list of BGR colors
    mask_colors = [colors(idx, True) for idx in class_indices]

    # Pass the *pre-resized* masks and the list of colors
    annotator.masks(
        resized_masks_np, 
        colors=mask_colors,
        alpha=0.5
    )

for i, box in enumerate(results.boxes):
    class_id = int(box.cls)
    label = f"{model.names[class_id]} {box.conf.item():.2f}"
    color = colors(class_id, True) # Get color for this class
    annotator.box_label(box.xyxy.squeeze(), label, color=color)

    #centorid logic
    if class_id == WEED_CLASS_ID and results.masks is not None:
        if i < len(results.masks.xy):
            mask = results.masks.xy[i]
            if len(mask) > 0:
                M = cv2.moments(mask.astype(np.int32))
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(annotator.im, (cX, cY), 5, (255, 255, 0), -1)

overlay_image = annotator.result()


overlay_image_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)

cv2.imwrite(OUTPUT_SAVE_PATH, overlay_image_bgr)
print(f"Successfully saved production overlay image to {OUTPUT_SAVE_PATH}")

# raw data
# for result in results:
#     if result.masks:
#         print("Masks found:")
#         for i, mask in enumerate(result.masks.xy):
#             class_id = int(result.boxes.cls[i])
#             class_name = model.names[class_id]
#             print(f"  - Object {i}: Class '{class_name}' ({class_id})")
#             # 'mask' is a numpy array of (x, y) points