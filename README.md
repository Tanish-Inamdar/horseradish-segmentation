Horseradish Segmentation Project 

This project is all about using deep learning to automatically identify horseradish plants and weeds from drone imagery. The goal is to build a segmentation model that can tell the difference between the crop and everything else on a pixel-by-pixel basis.

The model uses a powerful pre-trained DINOv3 backbone and a custom segmentation head built with PyTorch.

Files

    training.py: The main script for training the model. It loads the data, kicks off the training loop, and saves the best model checkpoints.

    evaluation.py: Used to check how well the model is doing. It calculates metrics like the Dice score and Mean IoU on the validation set.

    inference.py: A script to run a single image through the trained model and see the segmentation overlay. Perfect for quick tests and demos.

    segmentationDataset.py: A custom PyTorch Dataset class to load the horseradish images and their corresponding polygon label files.

    model.py: Defines the DinoV3ForSegmentation neural network architecture.

Getting Started on a New Machine

Here’s how to get this project running on a new computer with an NVIDIA GPU.

1. Clone the Repository

git clone https://github.com/Tanish-Inamdar/horseradish-segmentation.git
cd horseradish-segmentation

2. Set Up a Python Virtual Environment

# Create a virtual environment
python -m venv .venv

# Activate it (on Windows)
.\.venv\Scripts\activate

3. Install Dependencies


pip install -r requirements.txt

4. Make Sure to Have PyTorch

# First, uninstall any existing CPU-only version
pip uninstall torch torchvision torchaudio

# Then, install the CUDA-enabled version
# (This command is for CUDA 12.1, check the PyTorch website if you need a different one)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

5. Make Sure to Have the Dataset and Update DATA_DIR

The dataset isn't stored in Git. You'll need to download it to the lab computer and update the paths in the scripts.

    Copy your horseradish_dataset folder to the new machine.

    Open training.py and evaluation.py and update the path variables at the top of each file to point to the correct location of your dataset.

How to Run

Make sure your virtual environment is active before running any scripts!

To train the model:
    
    python training.py

    (Remember to increase the EPOCHS, BATCH_SIZE, and NUM_WORKERS in the script):
    BATCH_SIZE: 32, NUM_WORKERS: 4

To evaluate the best model:

    python evaluation.py

To see a prediction on a single image:

    python inference.py

