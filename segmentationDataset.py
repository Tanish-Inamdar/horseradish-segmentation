import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class HorseradishSegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset for horseradish and weed segmentation.

    Args:
        root_dir (str): The root directory of the dataset (e.g., 'horseradish_dataset/train/').
        processor (AutoImageProcessor): The image processor from Hugging Face for resizing and normalization.
    """
    def __init__(self, root_dir: str, processor):
        self.root_dir = root_dir
        self.processor = processor
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        
        # Get a sorted list of image filenames to ensure consistency
        self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])

        # Define the mapping from the class index in your .txt files to the mask value
        # Assume class 0 in your files is horseradish and class 1 is weed.
        self.class_map = {0: 1, 1: 2} # file_class_0 -> mask_value_1 (horseradish), file_class_1 -> mask_value_2 (weed)

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Loads and returns a single sample (image and mask) from the dataset.
        """
        # 1. Construct file paths
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        
        # Derive the label path from the image path
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)

        # 2. Load the image
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size

        # 3. Create the ground-truth mask from the label file
        # Start with a blank mask, where all pixels are 0 (background)
        mask = np.zeros((original_height, original_width), dtype=np.uint8)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    
                    # Convert normalized polygon coordinates to absolute pixel coordinates
                    polygon_normalized = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                    polygon_pixels = polygon_normalized * np.array([original_width, original_height])
                    
                    # Get the correct value to fill the mask with (1 for horseradish, 2 for weed)
                    mask_value = self.class_map.get(class_id, 0) # Default to 0 if class_id is unexpected
                    
                    # Draw the filled polygon onto the mask
                    from skimage.draw import polygon
                    rr, cc = polygon(polygon_pixels[:, 1], polygon_pixels[:, 0], mask.shape)
                    mask[rr, cc] = mask_value

        # 4. Preprocess the image and mask
        # We need to resize both the image and the mask to what the model expects.
        # The processor handles image resizing and normalization.
        inputs = self.processor(images=image, return_tensors="pt")
        
        # For the mask, we need to resize it manually and ensure it's a tensor.
        # We use "NEAREST" interpolation to avoid creating new pixel values (like 1.5).
        mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.processor.size['height'], interpolation=transforms.InterpolationMode.NEAREST)
        ])
        mask_tensor = mask_transform(mask)
        
        # Squeeze the channel dimension from the mask tensor (from [1, H, W] to [H, W])
        mask_tensor = mask_tensor.squeeze(0).long()

        return {
            "pixel_values": inputs['pixel_values'].squeeze(0), # Remove the batch dimension added by processor
            "labels": mask_tensor
        }