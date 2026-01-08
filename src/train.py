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
        sequence_len=cfg['data']['sequence_len'],
        split='train'
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
        dropout=cfg['model']['dropout'],
        img_size=tuple(cfg['data']['image_size'])
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
    
    # Scaler for AMP
    scaler = torch.amp.GradScaler('cuda')
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['train']['epochs'],
        eta_min=1e-6
    )
    
    # Training Loop
    print("Starting training...")
    best_loss = float('inf')
    save_dir = os.path.join(os.getcwd(), "runs", cfg['experiment']['name'], "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    # Resume logic
    start_epoch = 0
    loss_history = []
    acc_history = []
    resume_path = os.path.join(save_dir, "last.pth")
    
    if os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'loss_history' in checkpoint:
            loss_history = checkpoint['loss_history']
        if 'acc_history' in checkpoint:
            acc_history = checkpoint['acc_history']
        print(f"Resuming from epoch {start_epoch}")

    # Save resolved config
    os.makedirs(os.path.dirname(os.path.join("runs", cfg['experiment']['name'], "config_resolved.yaml")), exist_ok=True)
    with open(os.path.join("runs", cfg['experiment']['name'], "config_resolved.yaml"), 'w') as f:
        yaml.dump(cfg, f)
        
    for epoch in range(start_epoch, cfg['train']['epochs']):
        model.train()
        temporal_pool.train()
        
        total_loss = 0
        total_ce = 0
        total_tri = 0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        
        for batch_idx, (frames, labels) in enumerate(pbar):
            # frames: [B, T, C, H, W]
            frames = frames.to(device)
            labels = labels.to(device)
            
            B, T, C, H, W = frames.shape
            
            # Flatten for backbone: [B*T, C, H, W]
            frames_flat = frames.view(B*T, C, H, W)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                # Forward backbone
                features_flat, cls_scores_flat = model(frames_flat, return_logits=True) # [B*T, d], [B*T, num_classes]
                
                # Reshape features: [B, T, d]
                features = features_flat.view(B, T, -1)
                
                # Temporal Pooling
                seq_embedding = temporal_pool(features) # [B, d]
                
                cls_scores = cls_scores_flat.view(B, T, -1).mean(dim=1) # [B, num_classes]
                
                # Calculate Losses
                loss_ce = criterion_ce(cls_scores, labels)
                loss_tri = criterion_tri(seq_embedding, labels)
                
                loss = cfg['loss']['ce']['weight'] * loss_ce + cfg['loss']['triplet']['weight'] * loss_tri
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Calculate Accuracy
            _, preds = torch.max(cls_scores, 1)
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)
            
            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_tri += loss_tri.item()
            
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'ce': total_ce / (batch_idx + 1),
                'tri': total_tri / (batch_idx + 1),
                'acc': total_correct / total_samples,
                'lr': optimizer.param_groups[0]['lr']
            })
            
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_samples
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")
        
        loss_history.append(avg_loss)
        acc_history.append(avg_acc)
        
        # Save checkpoint
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': avg_loss,
            'loss_history': loss_history,
            'acc_history': acc_history
        }, os.path.join(save_dir, "last.pth"))
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
                'loss_history': loss_history,
                'acc_history': acc_history
            }, os.path.join(save_dir, "best.pth"))
            print(f"New best model saved with loss {best_loss:.4f}")
            
        # Plotting
        if (epoch + 1) % cfg['train'].get('plot_freq', 10) == 0:
            try:
                import matplotlib.pyplot as plt
                
                # Loss Plot
                plt.figure()
                plt.plot(loss_history)
                plt.title('Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.savefig(os.path.join(save_dir, 'loss.png'))
                plt.close()
                
                # Accuracy Plot
                plt.figure()
                plt.plot(acc_history)
                plt.title('Training Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.savefig(os.path.join(save_dir, 'accuracy.png'))
                plt.close()
            except Exception as e:
                print(f"Error plotting: {e}")

if __name__ == "__main__":
    main()
