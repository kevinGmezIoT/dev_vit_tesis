import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_base import load_config
from preprocess.dataset import MARSDataset
from preprocess.transforms import get_transforms
from model.vit_embedder import ViTEmbedder
from model.cnn_embedder import CNNEmbedder
from model.temporal_pooling import TemporalPooling

def extract_features_selective(model, temporal_pool, dataset, indices, device):
    model.eval()
    temporal_pool.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Extracting features"):
            frames, label = dataset[idx]
            frames = frames.unsqueeze(0).to(device)
            B, T, C, H, W = frames.shape
            
            features_flat = model(frames.view(B*T, C, H, W))
            features = temporal_pool(features_flat.view(B, T, -1))
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(label)
            
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
    return np.concatenate(all_features), np.array(all_labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-samples", type=int, default=1000, help="Limit samples for faster TSNE")
    parser.add_argument("--force-cpu", action="store_true", help="Do not use GPU")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    if torch.cuda.is_available() and not args.force_cpu:
        torch.cuda.empty_cache()
        
    if args.force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset
    transforms = get_transforms(tuple(cfg['data']['image_size']), is_train=False)
    dataset = MARSDataset(
        index_csv=cfg['data']['index_csv'],
        root_dir=cfg['data']['root'],
        transform=transforms,
        sequence_len=cfg['data']['sequence_len'],
        split='test'
    )
    
    # Subset if needed
    indices = list(range(len(dataset)))
    if len(indices) > args.max_samples:
        np.random.seed(42)
        indices = list(np.random.choice(indices, args.max_samples, replace=False))
    
    # Get num_classes from train split
    train_dataset = MARSDataset(cfg['data']['index_csv'], cfg['data']['root'], split='train')
    num_classes = train_dataset.num_classes

    # Model
    if 'vit' in cfg['experiment']['name'].lower():
        model = ViTEmbedder(cfg['model']['backbone'], embed_dim=cfg['model']['embed_dim'], num_classes=num_classes, img_size=tuple(cfg['data']['image_size']))
    else:
        model = CNNEmbedder(cfg['model']['backbone'], embed_dim=cfg['model']['embed_dim'], num_classes=num_classes)
    
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    
    try:
        model = model.to(device)
        print(f"Model loaded to {device}")
    except:
        print("Warning: GPU memory full. Falling back to CPU.")
        device = torch.device("cpu")
        model = model.to(device)
        
    temporal_pool = TemporalPooling().to(device)
    
    features, labels = extract_features_selective(model, temporal_pool, dataset, indices, device)
    
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap('tab20')
    
    for i, label in enumerate(unique_labels):
        if i >= 20: break 
        idx = (labels == label)
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=f"ID {label}", alpha=0.7)
    
    idx_remaining = np.isin(labels, unique_labels[20:], invert=False)
    if np.any(idx_remaining):
        plt.scatter(features_2d[idx_remaining, 0], features_2d[idx_remaining, 1], c='gray', alpha=0.1, label='Other IDs')

    plt.title(f"t-SNE Visualization - {cfg['experiment']['name']}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    
    output_path = os.path.join("outputs", cfg['experiment']['name'], "tsne_visualization.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main()
