import torch
import torch.nn as nn
from transformers import AutoModel

class DinoV3ForSegmentation(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super(DinoV3ForSegmentation, self).__init__()
        
        self.backbone = AutoModel.from_pretrained(model_name)
        
        self.freeze_backbone()
        
        # Get the feature dimension from the backbone's configuration
        # self.feature_dims = [96, 192, 384, 768] # For convnext tiny
        self.feature_dims = [128, 256, 512, 1024] # For convnext base
        #self.feature_dims = [192, 384, 768, 1536] # For convnext large
        
        # 2. Define the Decoder
        # This head will upsample the features from 1/32 of the image size back to full size.
        # 2^5 = 32, so we need 5 upsampling stages.
        self.decoder_head = nn.Sequential(
            # First, project the high-dimensional features to a smaller dimension
            nn.Conv2d(sum(self.feature_dims), 256, kernel_size=1),
            nn.ReLU(inplace=True),
            # Upsample 1: 56x56 -> 112x112
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Upsample 2: 112x112 -> 224x224
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # Final convolution to get the logits for each class
            nn.Conv2d(128, num_classes, kernel_size=3, padding=1)
        )

    def forward(self, pixel_values):
         ### FOR VIT MODEL TRAINING ###

        # features_seq = outputs.last_hidden_state
        # B, N, C = features_seq.shape
        
        # features_patches = features_seq[:, 1:, :]
        
        # H = W = int((N - 1) ** 0.5)
        
        # # Reshape the sequence of patches into a 2D feature map (B, C, H, W).
        # features_2d = features_patches.permute(0, 2, 1).reshape(B, C, H, W)
        
        ### FOR VIT MODEL TRAINING ###

        # Get outputs from the backbone
        outputs = self.backbone(
            pixel_values=pixel_values, 
            output_hidden_states = True,
            return_dict = True
        )

        hidden_states = [outputs.hidden_states[i] for i in [1, 2, 3, 4]]
        target_size = hidden_states[0].shape[-2:]
        upsampled_features = [
            nn.functional.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            for feat in hidden_states
        ]
        features_2d = torch.cat(upsampled_features, dim=1)
        segmentation_logits = self.decoder_head(features_2d)
        final_logits = nn.functional.interpolate(
            segmentation_logits,
            size=pixel_values.shape[-2:], # Match the input image size (e.g., 224x224)
            mode='bilinear',
            align_corners=False
        )        
        return final_logits, features2d

    def freeze_backbone(self):
        print("Freezing DINOv3 backbone.")
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        print("Unfreezing DINOv3 backbone for fine-tuning.")
        for param in self.backbone.parameters():
            param.requires_grad = True