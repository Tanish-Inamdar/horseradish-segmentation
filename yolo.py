import os
import torch
import lightly_train
from ultralytics import YOLO

TEACHER_MODEL_CLASS = "model.py:DinoV3ForSegmentation"
TEACHER_WEIGHTS_PATH = "/home/tanishi2/ag group/horseradish-segmentation/weights/model_best.pt"
TEACHER_MODEL_NAME = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
NUM_CLASSES = 3
STUDENT_MODEL_YAML = "ultralytics/yolov8n-seg.yaml"

DATASET_YAML_CONFIG = "/home/tanishi2/ag group/horseradish-segmentation/WeedDataset.yaml" # KEEP FOR REFERENCE
DATASET_ROOT_PATH = "/home/tanishi2/ag group/dataset" # NEW
#home: 
# DATASET_YAML_PATH = "/home/tanishi2/ag group/dataset/train/data.yaml"

BATCH_SIZE = 6
NUM_WORKERS = 3
OUTPUT_DIR = "out/yolov8n_dinov3_distilled" 
EXPERIMENT_NAME = "horseradish_distillation_v1"

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"distiallation training")
    print(f"  Teacher Model: {TEACHER_MODEL_CLASS}")
    print(f"  Teacher Weights: {TEACHER_WEIGHTS_PATH}")
    print(f"  Student Model: {STUDENT_MODEL_YAML}")
    print(f"  Dataset: {DATASET_ROOT_PATH}")
    
    lightly_train.train(
        out=OUTPUT_DIR,
        # experiment_name=EXPERIMENT_NAME,
        data=DATASET_ROOT_PATH,
        model=STUDENT_MODEL_YAML,
        trainer_args={
            "max_epochs": 100,
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
        },
        method="distillation",
        method_args={
            "teacher": TEACHER_MODEL_CLASS,
            "teacher_weights": TEACHER_WEIGHTS_PATH,
            "teacher_args": {
                "model_name": TEACHER_MODEL_NAME,
                "num_classes": NUM_CLASSES
            },
            
            "distill_loss_weight": 1.0, # should trust teacher model
            "task_loss_weight": 1.0,    # should trust annotated file 
        },
        loader_args={
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
        },
        overwrite=True
        #optimizer taken from Abhinav paper am not hypertuning currently try with lightly default
        # optimizer={ 
        #     "name": "AdamW",
        #     "lr": 1e-4,
        #     "weight_decay": 1e-5,
        # },
    )

    print(f"--- Distillation Complete! ---")
    print(f"Your trained YOLOv8 student model is saved in:")
    print(f"{OUTPUT_DIR}/{EXPERIMENT_NAME}/weights/best.pt")
    print(f"--------------------------------")
    trained_student_model = YOLO(f"{OUTPUT_DIR}/{EXPERIMENT_NAME}/weights/best.pt")
    
    #Run validation
    metrics = trained_student_model.val()
    print(metrics)

    #Run inference
    results = trained_student_model.predict("/path/to/new_image.jpg")
    print(results)