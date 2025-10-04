#!/usr/bin/env python3
"""
EfficientNet-B3 Model cho Leaf Disease Classification
- Multi-branch architecture cho shape, texture, color features
- Enhanced cho đặc trưng lá cây: hình dạng, đốm bệnh, gân lá
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Tuple

class EfficientNetLeafClassifier(nn.Module):
    """
    EfficientNet-B3 với multi-branch architecture cho leaf classification
    """
    
    def __init__(self, num_classes: int, num_disease_classes: int = None):
        super().__init__()
        
        # EfficientNet-B3 backbone
        self.backbone = timm.create_model(
            'efficientnet_b3', 
            pretrained=True,
            num_classes=0,  # Remove classifier
            global_pool=''  # Keep spatial dimensions
        )
        
        # Get feature dimensions
        self.feature_dim = self.backbone.num_features  # 1536 for B3
        self.num_classes = num_classes
        self.num_disease_classes = num_disease_classes
        
        # Multi-branch feature extraction
        self.shape_branch = self._create_branch("shape")
        self.texture_branch = self._create_branch("texture") 
        self.color_branch = self._create_branch("color")
        
        # Species classifier - Fix dimension mismatch
        # Each branch outputs 256 features, so 3 branches = 768 total
        self.species_classifier = nn.Sequential(
            nn.Linear(768, 1024),  # 256 * 3 = 768
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Disease classifier (if specified) - Fix dimension mismatch
        if num_disease_classes:
            self.disease_classifier = nn.Sequential(
                nn.Linear(768, 1024),  # 256 * 3 = 768
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_disease_classes)
            )
    
    def _create_branch(self, branch_type: str) -> nn.Module:
        """Tạo branch cho từng loại đặc trưng"""
        if branch_type == "shape":
            # Focus on edges, contours, shape features
            return nn.Sequential(
                nn.Conv2d(self.feature_dim, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
        elif branch_type == "texture":
            # Focus on texture, spots, surface patterns
            return nn.Sequential(
                nn.Conv2d(self.feature_dim, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
        else:  # color
            # Focus on color patterns, disease spots
            return nn.Sequential(
                nn.Conv2d(self.feature_dim, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """Forward pass"""
        # Extract backbone features
        features = self.backbone.forward_features(x)  # [B, 1536, H, W]
        
        # Multi-branch processing
        shape_feat = self.shape_branch(features).flatten(1)  # [B, 256]
        texture_feat = self.texture_branch(features).flatten(1)  # [B, 256]
        color_feat = self.color_branch(features).flatten(1)  # [B, 256]
        
        # Combine features
        combined = torch.cat([shape_feat, texture_feat, color_feat], dim=1)  # [B, 768]
        
        # Species prediction
        species_logits = self.species_classifier(combined)
        
        if return_features:
            return species_logits, combined
        else:
            return species_logits
    
    def forward_disease(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for disease classification"""
        if not hasattr(self, 'disease_classifier'):
            raise ValueError("Disease classifier not initialized")
        
        # Extract features
        _, combined_features = self.forward(x, return_features=True)
        
        # Disease prediction
        disease_logits = self.disease_classifier(combined_features)
        return disease_logits


class EfficientNetDiseaseClassifier(nn.Module):
    """
    EfficientNet-B3 cho disease classification cho từng species
    """
    
    def __init__(self, num_disease_classes: int):
        super().__init__()
        
        # EfficientNet-B3 backbone
        self.backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        
        self.feature_dim = self.backbone.num_features
        self.num_disease_classes = num_disease_classes
        
        # Disease-specific feature extraction
        self.disease_branch = nn.Sequential(
            nn.Conv2d(self.feature_dim, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Disease classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_disease_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.backbone.forward_features(x)
        disease_feat = self.disease_branch(features).flatten(1)
        logits = self.classifier(disease_feat)
        return logits


def create_species_model(num_classes: int) -> EfficientNetLeafClassifier:
    """Tạo species classifier"""
    return EfficientNetLeafClassifier(num_classes=num_classes)


def create_disease_model(num_disease_classes: int) -> EfficientNetDiseaseClassifier:
    """Tạo disease classifier"""
    return EfficientNetDiseaseClassifier(num_disease_classes=num_disease_classes)
