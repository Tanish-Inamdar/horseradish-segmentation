import os
import torch
import numpy as np
import cv2 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

class HorseradishSegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset for horseradish and weed segmentation.
    """
    def __init__(self, root_dir: str, processor):
        self.root_dir = root_dir
        self.processor = processor
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.transform = A.Compose([
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.RandomRotate90(p=0.5),
            # A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.3),
            # A.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.1, rotate_limit=10),
            # A.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
            # A.GaussNoise(p=0.1),
            # A.ElasticTransform(p=0.1, alpha=120, sigma=120 * 0.05),
            # A.MotionBlur(blur_limit=[13, 17], allow_shifted=False, angle_range=[0, 0], direction_range=[0, 0]),
            # A.Downscale(scale_range=[0.25, 0.25], interpolation_pair={"upscale":0,"downscale":0}),
        ])
        self.processor = processor
        self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        self.class_map = {0: 1, 1: 2} # file_class_0 -> mask_value_1 (horseradish), file_class_1 -> mask_value_2 (weed)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name) #
        image_pil = Image.open(image_path).convert("RGB")
        original_width, original_height = image_pil.size
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

        image_np = np.array(image_pil)

        if self.transform:
             augmented = self.transform(image=image_np, mask=mask)
             image_np = augmented['image']
             mask = augmented['mask'] 
        # --- Manually transform the image and mask separately ---

        # 1. Process the image using the Hugging Face processor
        inputs = self.processor(images=image_np, return_tensors="pt") 

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