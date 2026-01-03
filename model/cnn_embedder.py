import torch
import torch.nn as nn
import timm

class CNNEmbedder(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True, embed_dim=512, num_classes=None, dropout=0.1):
        super(CNNEmbedder, self).__init__()
        
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool=''   # We'll do our own pooling if needed, or use timm's default
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 128)
            features = self.backbone(dummy_input)
            if len(features.shape) == 4:
                # If it's B, C, H, W, we do global pool
                self.feature_dim = features.shape[1]
                self.global_pool = nn.AdaptiveAvgPool2d(1)
            else:
                self.feature_dim = features.shape[-1]
                self.global_pool = nn.Identity()

        self.dropout = nn.Dropout(p=dropout)
        self.embedding_layer = nn.Linear(self.feature_dim, embed_dim)
        
        if num_classes is not None:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = None

    def forward(self, x, return_logits=False):
        # x: [B, C, H, W]
        features = self.backbone(x)
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        
        embedding = self.embedding_layer(self.dropout(features))
        
        if return_logits and self.classifier is not None:
            logits = self.classifier(embedding)
            return embedding, logits
        
        return embedding
