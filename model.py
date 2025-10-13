import torch
import torch.nn as nn
from transformers import AutoModel

class DinoV3ForSegmentation(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super(DinoV3ForSegmentation, self).__init__()
        
        self.backbone = AutoModel.from_pretrained(model_name)
        
        self.freeze_backbone()
        
        # Get the feature dimension from the backbone's configuration
        backbone_dim = self.backbone.config.hidden_sizes[-1] # For ConvNeXt, this is the final feature dimension

        # 2. Define the Segmentation Head (Decoder)
        # This head will upsample the features from 1/32 of the image size back to full size.
        # 2^5 = 32, so we need 5 upsampling stages.
        self.segmentation_head = nn.Sequential(
            # First, project the high-dimensional features to a smaller dimension
            nn.Conv2d(backbone_dim, 256, kernel_size=1),
            
            # Upsample 1: 1/32 -> 1/16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Upsample 2: 1/16 -> 1/8
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Upsample 3: 1/8 -> 1/4
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Upsample 4: 1/4 -> 1/2
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Upsample 5: 1/2 -> 1/1 (Full Resolution)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # Final convolution to get the logits for each class
            nn.Conv2d(16, num_classes, kernel_size=3, padding=1)
        )

    def forward(self, pixel_values):
        """
        Forward pass of the model.
        """
        # Get outputs from the backbone
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states = True, return_dict = True)

        features_2d = outputs.hidden_states[-1]

        ### FOR VIT MODEL TRAINING ###

        # features_seq = outputs.last_hidden_state
        # B, N, C = features_seq.shape
        
        # features_patches = features_seq[:, 1:, :]
        
        # H = W = int((N - 1) ** 0.5)
        
        # # Reshape the sequence of patches into a 2D feature map (B, C, H, W).
        # features_2d = features_patches.permute(0, 2, 1).reshape(B, C, H, W)
        
        ### FOR VIT MODEL TRAINING ###

        segmentation_logits = self.segmentation_head(features_2d)
        
        return segmentation_logits

    def freeze_backbone(self):
        print("Freezing DINOv3 backbone.")
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        print("Unfreezing DINOv3 backbone for fine-tuning.")
        for param in self.backbone.parameters():
            param.requires_grad = True