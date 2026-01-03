import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ViTEmbedder(nn.Module):
    def __init__(self, backbone_name, pretrained=True, embed_dim=512, num_classes=0, dropout=0.0, img_size=None):
        super().__init__()
        
        # Create backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        
        # Get feature dim
        if hasattr(self.backbone, 'num_features'):
            self.in_features = self.backbone.num_features
        else:
            # Fallback for some models
            dummy = torch.randn(1, 3, 224, 224)
            out = self.backbone(dummy)
            self.in_features = out.shape[1]
            
        # Projection head
        self.bottleneck = nn.BatchNorm1d(self.in_features)
        self.bottleneck.bias.requires_grad_(False) # no shift
        self.bottleneck.apply(self._weights_init_kaiming)
        
        self.classifier = nn.Linear(self.in_features, num_classes, bias=False)
        self.classifier.apply(self._weights_init_classifier)
        
    def _weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, return_logits=False):
        # x: [B, C, H, W]
        features = self.backbone(x) # [B, dim]
        
        bn_features = self.bottleneck(features)
        
        if return_logits:
            cls_score = self.classifier(bn_features)
            return features, cls_score
        
        return features
