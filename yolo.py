import lightly_train
from ultralytics import settings

#i don't belive useful for my own data
# from ultralytics.data.utils import check_det_dataset
# dataset = check_det_dataset=("/home/tanishi2/ag group/horseradish-segmentation/WeedDataset.yaml")

data_path  = f"{settings['datasets_dir']}/home/tanishi2/ag group/dataset/train"




# TO DO: 
# Compare my model weight(if this even works w/ dinov3) with meta weights
# 
#

if __name__ == "__main__":
    # Distill the pretrained DINOv2 model to a ResNet-18 student model.
    lightly_train.train(
        out="out/my_distillation_pretrain_experiment",
        data=data_path,
        model="ultralytics/yolov8n-seg.yaml",
        method="distillation",
        method_args={
            "teacher": "dinov3/vitb16",
            "teacher_weights": "/home/tanishi2/ag group/horseradish-segmentation/weights/model_best.pt", # pretrained `dinov2/vitb14` weights 
        }
    )

    model = YOLO("/home/tanishi2/ag group/horseradish-segmentation/weights/dinov3_model_trained.pt")
    model.train(data_path, epochs=100)