import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import json

from preprocess.dataset import MARSDataset, RandomIdentitySampler
from preprocess.transforms import get_transforms
from model.temporal_pooling import TemporalPooling
from model.losses import CrossEntropyLabelSmooth, TripletLoss

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_model(cfg, model, device):
    # Data
    print("Loading data...")
    train_transforms = get_transforms(tuple(cfg['data']['image_size']), is_train=True)
    val_transforms = get_transforms(tuple(cfg['data']['image_size']), is_train=False)
    
    train_dataset = MARSDataset(
        index_csv=cfg['data']['index_csv'],
        root_dir=cfg['data']['root'],
        transform=train_transforms,
        sequence_len=cfg['data']['sequence_len'],
        split='train'
    )
    
    val_dataset = MARSDataset(
        index_csv=cfg['data']['index_csv'],
        root_dir=cfg['data']['root'],
        transform=val_transforms,
        sequence_len=cfg['data']['sequence_len'],
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['train']['batch']['P'] * cfg['train']['batch']['K'],
        sampler=RandomIdentitySampler(train_dataset, cfg['train']['batch']['P'] * cfg['train']['batch']['K'], cfg['train']['batch']['K']),
        num_workers=cfg['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['train']['batch']['P'] * cfg['train']['batch']['K'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True
    )
    
    # Model
    model = model.to(device)
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
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['train']['epochs'],
        eta_min=1e-6
    )
    
    # Training Loop
    print(f"Starting training for {cfg['experiment']['name']}...")
    best_val_loss = float('inf')
    output_dir = os.path.join("outputs", cfg['experiment']['name'])
    save_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(cfg['train']['epochs']):
        model.train()
        temporal_pool.train()
        
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [Train]")
        for frames, labels in pbar:
            frames, labels = frames.to(device), labels.to(device)
            B, T, C, H, W = frames.shape
            frames_flat = frames.view(B*T, C, H, W)
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    feat_flat, logits_flat = model(frames_flat, return_logits=True)
                    embedding = temporal_pool(feat_flat.view(B, T, -1))
                    logits = logits_flat.view(B, T, -1).mean(dim=1)
                    
                    loss_ce = criterion_ce(logits, labels)
                    loss_tri = criterion_tri(embedding, labels)
                    loss = cfg['loss']['ce']['weight'] * loss_ce + cfg['loss']['triplet']['weight'] * loss_tri
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                feat_flat, logits_flat = model(frames_flat, return_logits=True)
                embedding = temporal_pool(feat_flat.view(B, T, -1))
                logits = logits_flat.view(B, T, -1).mean(dim=1)
                
                loss_ce = criterion_ce(logits, labels)
                loss_tri = criterion_tri(embedding, labels)
                loss = cfg['loss']['ce']['weight'] * loss_ce + cfg['loss']['triplet']['weight'] * loss_tri
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(logits, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({'loss': train_loss / (pbar.n + 1), 'acc': train_correct / train_total})
            
        scheduler.step()
        
        # Validation
        model.eval()
        temporal_pool.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for frames, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [Val]"):
                frames, labels = frames.to(device), labels.to(device)
                B, T, C, H, W = frames.shape
                frames_flat = frames.view(B*T, C, H, W)
                
                feat_flat, logits_flat = model(frames_flat, return_logits=True)
                embedding = temporal_pool(feat_flat.view(B, T, -1))
                logits = logits_flat.view(B, T, -1).mean(dim=1)
                
                loss_ce = criterion_ce(logits, labels)
                loss_tri = criterion_tri(embedding, labels)
                loss = cfg['loss']['ce']['weight'] * loss_ce + cfg['loss']['triplet']['weight'] * loss_tri
                
                val_loss += loss.item()
                _, preds = torch.max(logits, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Train Acc {avg_train_acc:.4f}, Val Loss {avg_val_loss:.4f}, Val Acc {avg_val_acc:.4f}")
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        # Save last checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
        }, os.path.join(save_dir, "last.pth"))
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'history': history,
            }, os.path.join(save_dir, "best.pth"))
            print(f"Best model saved with validation loss {best_val_loss:.4f}")
            
    # Save history to JSON
    with open(os.path.join(output_dir, "history.json"), 'w') as f:
        json.dump(history, f)
        
    return history
