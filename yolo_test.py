import os
import random
import time
import cv2
import glob
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# --- CONFIGURATION ---
# Path to your trained weights
MODEL_PATH = "/home/tanishi2/ag group/horseradish-segmentation/YOLO_Distilled_Training/yolov8m_from_dinov3_teacher200epochs/weights/best.pt"

# Path to your validation images (Check your WeedDataset.yaml path!)
# Based on your YAML, it should be here:
VAL_IMAGES_PATH = "/home/tanishi2/ag group/dataset/val/images" 

OUTPUT_DIR = "inference_results_batch"
NUM_TEST_IMAGES = 10  # How many random images to save
SPEED_TEST_LOOPS = 100  # How many times to run inference for the speed test

def run_speed_benchmark(model, image_path, device):
    """
    Measures pure inference speed (including pre/post-processing).
    """
    print(f"\n--- ⏱️ Starting Speed Benchmark on {device} ---")
    
    # Load one image for testing
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image for speed test.")
        return

    # 1. WARMUP
    # GPU needs a few dummy runs to initialize CUDA context fully
    print("Warming up GPU...")
    for _ in range(10):
        _ = model(img, verbose=False)

    # 2. BENCHMARK
    print(f"Running {SPEED_TEST_LOOPS} inference loops...")
    start_time = time.time()
    
    for _ in range(SPEED_TEST_LOOPS):
        _ = model(img, verbose=False)
        
    end_time = time.time()
    
    # 3. CALCULATE
    total_time = end_time - start_time
    avg_time_per_img = total_time / SPEED_TEST_LOOPS
    fps = 1 / avg_time_per_img
    
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Latency: {avg_time_per_img * 1000:.2f} ms")
    print(f"⚡ Estimated FPS: {fps:.2f} frames/second")
    
    if fps < 10:
        print("⚠️ NOTE: For real-time robotics, we usually aim for >10-15 FPS.")
    else:
        print("✅ Speed looks good for real-time deployment!")

def run_random_visualization(model, image_files):
    """
    Picks random images, runs inference, and saves the overlay.
    """
    print(f"\n--- 👁️ Generating Visuals for {NUM_TEST_IMAGES} Random Images ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Pick random samples
    # Ensure we don't try to pick more than exist
    sample_size = min(len(image_files), NUM_TEST_IMAGES)
    test_files = random.sample(image_files, sample_size)
    
    for i, img_path in enumerate(test_files):
        filename = os.path.basename(img_path)
        
        # Run inference
        # save=True automatically saves to runs/segment/predict... 
        # but plotting manually gives us more control if we want custom overlays later.
        results = model(img_path, verbose=False)
        
        for r in results:
            # Plot the results (draws boxes and masks on the image)
            im_array = r.plot()  # plot() returns a BGR numpy array
            
            # Save to our specific output folder
            save_path = os.path.join(OUTPUT_DIR, f"pred_{filename}")
            cv2.imwrite(save_path, im_array)
            
    print(f"✅ Saved {sample_size} predictions to: {os.path.abspath(OUTPUT_DIR)}")

def main():
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        return

    # Load Model
    model = YOLO(MODEL_PATH)
    device = model.device
    print(f"Loaded Model: {MODEL_PATH}")
    print(f"Running on: {device}")

    # Find Images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG']
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(VAL_IMAGES_PATH, ext)))
    
    if not all_images:
        print(f"❌ No images found in {VAL_IMAGES_PATH}. Check your path!")
        return
    
    print(f"Found {len(all_images)} images in validation set.")

    # --- PART 1: VISUAL RANDOM CHECK ---
    run_random_visualization(model, all_images)

    # --- PART 2: SPEED TEST ---
    # Use the first image found for the speed test
    run_speed_benchmark(model, all_images[0], device)
    
    # --- PART 3: HARD METRICS (Optional but Recommended) ---
    print("\n--- 📊 Running Validation Metrics (mAP) ---")
    print("This calculates the official accuracy on the entire validation set.")
    # We use the dataset YAML you uploaded
    dataset_yaml = "/home/tanishi2/ag group/horseradish-segmentation/WeedDataset.yaml"
    
    if os.path.exists(dataset_yaml):
        # validation uses the 'val' split defined in your YAML
        metrics = model.val(data=dataset_yaml, split='val', verbose=False)
        print(f"mAP@50-95 (Segmentation): {metrics.seg.map:.4f}")
        print(f"mAP@50-95 (Box): {metrics.box.map:.4f}")
    else:
        print("Skipping metrics, YAML not found.")

if __name__ == "__main__":
    main()