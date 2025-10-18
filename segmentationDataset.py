import os
import torch
import numpy as np
import cv2 # Using OpenCV for robust polygon drawing
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

class HorseradishSegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset for horseradish and weed segmentation.
    """
    def __init__(self, root_dir: str, processor):
        self.root_dir = root_dir
        self.processor = processor
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        
        self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        self.class_map = {0: 1, 1: 2} # file_class_0 -> mask_value_1 (horseradish), file_class_1 -> mask_value_2 (weed)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)

        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size

        # --- Using NumPy and OpenCV for robust mask creation ---
        mask = np.zeros((original_height, original_width), dtype=np.uint8)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 3 or len(parts) % 2 == 0: continue # Skip malformed lines #<-- Keep this check
                    
                    class_id = int(parts[0])
                    mask_value = self.class_map.get(class_id, 0)
                    
                    # Denormalize polygon coordinates
                    polygon_normalized = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                    polygon_pixels = polygon_normalized * np.array([original_width, original_height])
                    polygon_pixels = polygon_pixels.astype(np.int32)
                    
                    # Draw the filled polygon onto the mask
                    cv2.fillPoly(mask, [polygon_pixels], color=mask_value)
        # --- Manually transform the image and mask separately ---

        # 1. Process the image using the Hugging Face processor
        inputs = self.processor(images=image, return_tensors="pt")

        # 2. Convert numpy mask directly to tensor WITHOUT scaling
        # Convert numpy array HxW to tensor HxW
        mask_tensor = torch.from_numpy(mask)

        # Add a channel dimension: HxW -> 1xHxW (required by Resize)
        mask_tensor = mask_tensor.unsqueeze(0)

        # Resize the tensor using functional interpolate (more direct than transforms.Resize)
        target_height = self.processor.size['height']
        target_width = self.processor.size['width'] # Assuming square, get width too
        mask_tensor = F.interpolate(
            mask_tensor.float().unsqueeze(0), # Add batch dim temporarily, ensure float for interpolate
            size=(target_height, target_width),
            mode='nearest' # Use nearest neighbor for masks
        ).squeeze(0).long() # Remove batch dim, convert back to long for loss function

        # Remove the channel dimension: 1xHxW -> HxW
        mask_tensor = mask_tensor.squeeze(0)

        return {
            "pixel_values": inputs['pixel_values'].squeeze(0),
            "labels": mask_tensor
        }