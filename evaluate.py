import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import json
import pandas as pd

from preprocess.dataset import MARSDataset
from preprocess.transforms import get_transforms
from model.vit_embedder import ViTEmbedder
from model.cnn_embedder import CNNEmbedder
from model.temporal_pooling import TemporalPooling
from utils.metrics import calculate_map, calculate_inter_intra_ratio

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--arch", type=str, choices=['vit', 'cnn'], required=True)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_transforms = get_transforms(tuple(cfg['data']['image_size']), is_train=False)
    
    # Load test dataset
    test_dataset = MARSDataset(
        index_csv=cfg['data']['index_csv'],
        root_dir=cfg['data']['root'],
        transform=test_transforms,
        sequence_len=cfg['data']['sequence_len'],
        split='test'
    )
    
    # Also need num_classes for initialization
    train_dataset = MARSDataset(cfg['data']['index_csv'], cfg['data']['root'], split='train')
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=cfg['data']['num_workers']
    )
    
    if args.arch == 'vit':
        model = ViTEmbedder(
            backbone_name=cfg['model']['backbone'],
            pretrained=False,
            embed_dim=cfg['model']['embed_dim'],
            num_classes=train_dataset.num_classes,
            img_size=tuple(cfg['data']['image_size'])
        )
    else:
        model = CNNEmbedder(
            backbone_name=cfg['model']['backbone'],
            pretrained=False,
            embed_dim=cfg['model']['embed_dim'],
            num_classes=train_dataset.num_classes
        )
        
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    temporal_pool = TemporalPooling().to(device)
    temporal_pool.eval()
    
    all_features = []
    all_pids = []
    all_camids = []
    
    print("Extracting features...")
    with torch.no_grad():
        for frames, labels in tqdm(test_loader):
            frames = frames.to(device)
            B, T, C, H, W = frames.shape
            frames_flat = frames.view(B*T, C, H, W)
            
            feat_flat = model(frames_flat)
            embedding = temporal_pool(feat_flat.view(B, T, -1))
            
            all_features.append(embedding.cpu())
            all_pids.extend(labels.tolist())
            
            # Camera IDs are in the CSV
            # To match accurately, we should pass them through the loader.
            # For simplicity in this script, let's assume we can retrieve them from the original df
            # but a better way is to include them in MARSDataset.__getitem__
    
    all_features = torch.cat(all_features, dim=0)
    
    # Re-map labels back to original PIDs and get camids for mAP
    pids = [test_dataset.pids[i] for i in all_pids]
    
    # Get camids from metadata (this matches the order of the dataset)
    camids = test_dataset.df['cam_id'].tolist()
    # Need to convert cam_id (e.g., 'c001') to int
    camids = [int(c.replace('c', '')) for c in camids]

    print("Calculating metrics...")
    # For mAP we'll use a simple query/gallery split (e.g., first 100 as query)
    # In a real Re-ID eval, this is more complex, but here we'll use the whole set 
    # and exclude same cam/same pid in calculate_map.
    
    mAP = calculate_map(all_features, pids, camids, all_features, pids, camids)
    ratio, inter, intra = calculate_inter_intra_ratio(all_features, torch.tensor(all_pids))
    
    results = {
        'mAP': mAP,
        'ratio_inter_intra': ratio,
        'avg_inter_dist': inter,
        'avg_intra_dist': intra
    }
    
    print(f"Results for {args.arch}:")
    print(json.dumps(results, indent=4))
    
    output_dir = os.path.join("outputs", cfg['experiment']['name'])
    with open(os.path.join(output_dir, "eval_results.json"), 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
