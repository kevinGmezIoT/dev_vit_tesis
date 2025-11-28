import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from preprocess.dataset import MARSDataset, RandomIdentitySampler
from preprocess.transforms import get_transforms
from model.vit_embedder import ViTEmbedder
from model.temporal_pooling import TemporalPooling
from model.losses import CrossEntropyLabelSmooth, TripletLoss

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Train Visual Feature Extractor")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    print("Loading data...")
    train_transforms = get_transforms(tuple(cfg['data']['image_size']), is_train=True)
    
    train_dataset = MARSDataset(
        index_csv=cfg['data']['index_csv'],
        root_dir=cfg['data']['root'],
        transform=train_transforms,
        sequence_len=cfg['data']['sequence_len']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['train']['batch']['P'] * cfg['train']['batch']['K'],
        sampler=RandomIdentitySampler(train_dataset, cfg['train']['batch']['P'] * cfg['train']['batch']['K'], cfg['train']['batch']['K']),
        num_workers=cfg['data']['num_workers'],
        pin_memory=True
    )
    
    # Model
    print("Building model...")
    model = ViTEmbedder(
        backbone_name=cfg['model']['backbone'],
        pretrained=cfg['model']['pretrained'],
        embed_dim=cfg['model']['embed_dim'],
        num_classes=train_dataset.num_classes,
        dropout=cfg['model']['dropout']
    ).to(device)
    
    temporal_pool = TemporalPooling().to(device)
    
    # Losses
    criterion_ce = CrossEntropyLabelSmooth(num_classes=train_dataset.num_classes, epsilon=cfg['loss']['ce']['label_smoothing'])
    criterion_tri = TripletLoss(margin=cfg['loss']['triplet']['margin'])
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['train']['optimizer']['lr'],
        weight_decay=cfg['train']['optimizer']['weight_decay']
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['train']['epochs'],
        eta_min=1e-6
    )
    
    # Training Loop
    print("Starting training...")
    best_loss = float('inf')
    save_dir = os.path.join("runs", cfg['experiment']['name'], "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save resolved config
    with open(os.path.join("runs", cfg['experiment']['name'], "config_resolved.yaml"), 'w') as f:
        yaml.dump(cfg, f)
        
    for epoch in range(cfg['train']['epochs']):
        model.train()
        temporal_pool.train()
        
        total_loss = 0
        total_ce = 0
        total_tri = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        
        for batch_idx, (frames, labels) in enumerate(pbar):
            # frames: [B, T, C, H, W]
            frames = frames.to(device)
            labels = labels.to(device)
            
            B, T, C, H, W = frames.shape
            
            # Flatten for backbone: [B*T, C, H, W]
            frames_flat = frames.view(B*T, C, H, W)
            
            # Forward backbone
            features_flat, cls_scores_flat = model(frames_flat, return_logits=True) # [B*T, d], [B*T, num_classes]
            
            # Reshape features: [B, T, d]
            features = features_flat.view(B, T, -1)
            
            # Temporal Pooling
            seq_embedding = temporal_pool(features) # [B, d]
            
            # For CE loss, we can use the average of cls_scores or just use the embedding to classify?
            # Usually in ReID with ViT, we use the CLS token of each frame or average them.
            # Here, the model returns cls_scores for each frame.
            # Let's average the cls_scores over time for the sequence prediction?
            # Or simpler: use the seq_embedding to predict class (requires another classifier layer).
            # BUT, the ViTEmbedder has a classifier layer that works on frame features.
            # Let's use the average of frame-level logits for CE loss.
            
            cls_scores = cls_scores_flat.view(B, T, -1).mean(dim=1) # [B, num_classes]
            
            # Calculate Losses
            loss_ce = criterion_ce(cls_scores, labels)
            loss_tri = criterion_tri(seq_embedding, labels)
            
            loss = cfg['loss']['ce']['weight'] * loss_ce + cfg['loss']['triplet']['weight'] * loss_tri
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_tri += loss_tri.item()
            
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'ce': total_ce / (batch_idx + 1),
                'tri': total_tri / (batch_idx + 1),
                'lr': optimizer.param_groups[0]['lr']
            })
            
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join(save_dir, "last.pth"))
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_dir, "best.pth"))
            print(f"New best model saved with loss {best_loss:.4f}")

if __name__ == "__main__":
    main()
