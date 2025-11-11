import lightly_train

if __name__ == "__main__":
    # Distill the pretrained DINOv2 model to a ResNet-18 student model.
    lightly_train.train(
        out="out/my_distillation_pretrain_experiment",
        data="home/tanishi2/ag group/dataset",
        model="ultralytics/yolov8m-seg.yaml",
        method="distillation",
        method_args={
            "teacher": "dinov3/vitb16",
            "teacher_weights": "/home/tanishi2/ag group/horseradish-segmentation/weights/model_best.pt", # pretrained `dinov2/vitb14` weights 
        }
    )