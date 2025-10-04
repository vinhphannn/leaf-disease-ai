#!/usr/bin/env python3
"""
EfficientNet-B0 model architecture to match the trained model
"""

import torch
import torch.nn as nn
import timm

class EfficientNetB0LeafClassifier(nn.Module):
    """EfficientNet-B0 based leaf disease classifier"""
    
    def __init__(self, num_classes=38):
        super().__init__()
        
        # Create EfficientNet-B0 backbone
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False)
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Classify
        output = self.classifier(features)
        
        return output

def create_efficientnet_b0_model(num_classes=38):
    """Create EfficientNet-B0 model"""
    return EfficientNetB0LeafClassifier(num_classes=num_classes)
