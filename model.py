import torch
import torch.nn as nn
from transformers import AutoModel

class DinoV3ForSegmentation(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super(DinoV3ForSegmentation, self).__init__()
        
        self.backbone = AutoModel.from_pretrained(model_name)
        
        self.freeze_backbone()
        
        # Get the feature dimension from the backbone's configuration
        # self.feature_dims = self.config.hidden_sizes
        # self.feature_dims = [96, 192, 384, 768] # For convnext tiny
        # self.feature_dims = [128, 256, 512, 1024] # For convnext base
        self.feature_dims = [192, 384, 768, 1536] # For convnext large
        
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
        return final_logits, features_2d #fix later jsut comment and uncomment :/
        # return final_logits

    # predicted_mask = torch.argmax(final_logits, dim=1).cpu().numpy()
    # WEED_CLASS_ID = 2  
    # weed_mask = (predicted_mask[0] == WEED_CLASS_ID).astype(np.uint8) * 255

    def freeze_backbone(self):
        print("Freezing DINOv3 backbone.")
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        print("Unfreezing DINOv3 backbone for fine-tuning.")
        for param in self.backbone.parameters():
            param.requires_grad = True


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import AutoModel, AutoConfig

# class DinoV3ForSegmentation(nn.Module):
#     def __init__(self, model_name: str, num_classes: int):
#         super(DinoV3ForSegmentation, self).__init__()
        
#         # 1. Load the backbone and its configuration
#         self.backbone = AutoModel.from_pretrained(model_name)
#         self.config = AutoConfig.from_pretrained(model_name)
        
#         # Automatically get feature dimensions from the config
#         # e.g., [128, 256, 512, 1024] for 'base'
#         self.feature_dims = self.config.hidden_sizes
        
#         # 2. Define a more efficient FPN-style (U-Net) decoder
#         # This is much more memory-friendly than concatenating 1920 channels
        
#         # Common dimension for all feature maps in the decoder
#         decoder_dim = 256 

#         # 1x1 convs to project each backbone stage to decoder_dim
#         self.lateral_convs = nn.ModuleList()
#         for dim in reversed(self.feature_dims):
#             self.lateral_convs.add_module(
#                 f"lateral_conv_{dim}", nn.Conv2d(dim, decoder_dim, kernel_size=1)
#             )

#         # 3x3 convs to refine the upsampled and summed features
#         self.output_convs = nn.ModuleList([
#                     nn.Sequential(
#                         nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1, bias=False),
#                         nn.BatchNorm2d(decoder_dim),
#                         nn.ReLU(inplace=True)
#                     )
#                     for _ in range(len(self.feature_dims) - 1)
#                 ])
#         # Final classifier head
#         self.decoder_head = nn.Sequential(
#             nn.Conv2d(decoder_dim, 128, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, num_classes, kernel_size=1)
#         )

#         # 3. Freeze the backbone initially
#         self.freeze_backbone()

#     def forward(self, pixel_values):
#         # 1. Get all hidden states from the backbone
#         outputs = self.backbone(
#             pixel_values=pixel_values, 
#             output_hidden_states=True,
#             return_dict=True
#         )

#         # Get the 4 stage outputs (hidden_states[1] to [4])
#         # These are at strides 4, 8, 16, 32
#         hidden_states = outputs.hidden_states[1:] 

#         # 2. Process features through the decoder (from deep to shallow)
#         lateral_features = []
#         for conv, state in zip(self.lateral_convs, reversed(hidden_states)):
#             lateral_features.append(conv(state))

#         # Start with the deepest feature (P4, stride 32)
#         p = lateral_features[0] 
        
#         # Progressively upsample and add
#         for i in range(len(self.output_convs)):
#             # Upsample previous feature map
#             p = F.interpolate(p, scale_factor=2, mode='bilinear', align_corners=False)
            
#             # Add lateral connection from the corresponding backbone stage
#             p = p + lateral_features[i+1]
            
#             # Refine with 3x3 conv
#             p = self.output_convs[i](p)

#         # 'p' is now the final, fused feature map at stride 4 (e.g., 56x56)
        
#         # 3. Classify and upsample to original image size
#         segmentation_logits = self.classifier(p)
        
#         final_logits = F.interpolate(
#             segmentation_logits,
#             size=pixel_values.shape[-2:], # Match the input image size
#             mode='bilinear',
#             align_corners=False
#         )        
        
#         # Return only the final logits as expected by your loss function
#         return final_logits, p # You can return 'p' for feature analysis if needed

#     def freeze_backbone(self):
#         print("Freezing DINOv3 backbone.")
#         for param in self.backbone.parameters():
#             param.requires_grad = False
            
#     def unfreeze_backbone(self):
#         print("Unfreezing DINOv3 backbone for fine-tuning.")
#         for param in self.backbone.parameters():
#             param.requires_grad = True