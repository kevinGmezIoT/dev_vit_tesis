import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from train_base import load_config
from preprocess.dataset import MARSDataset
from preprocess.transforms import get_transforms
from model.vit_embedder import ViTEmbedder
from model.cnn_embedder import CNNEmbedder
from model.temporal_pooling import TemporalPooling

def get_representative_frame(dataset, idx):
    frames_dir = dataset.df.iloc[idx]['frames_dir']
    full_dir = os.path.join(dataset.root_dir, frames_dir)
    imgs = sorted(os.listdir(full_dir))
    if not imgs: return None
    middle_idx = len(imgs) // 2
    return Image.open(os.path.join(full_dir, imgs[middle_idx]))

def extract_features_selective(model, temporal_pool, dataset, indices, device):
    model.eval()
    temporal_pool.eval()
    features = []
    
    # Process one by one - extremely safe for memory
    with torch.no_grad():
        for idx in tqdm(indices, desc="Extracting features for selected sequences"):
            # Manually get item from dataset
            frames, _ = dataset[idx]
            frames = frames.unsqueeze(0).to(device) # [1, T, C, H, W]
            
            B, T, C, H, W = frames.shape
            f_flat = model(frames.view(B*T, C, H, W))
            f_pool = temporal_pool(f_flat.view(B, T, -1))
            f_norm = torch.nn.functional.normalize(f_pool, p=2, dim=1)
            
            features.append(f_norm.cpu())
            
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
    return torch.cat(features, dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--query-ids", type=str, required=True, help="Comma-separated sequence_ids (e.g. ID0001_S0001_c001_cyc0001)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gallery-subset", type=int, default=1000, help="Limit gallery for memory safety")
    parser.add_argument("--force-cpu", action="store_true", help="Do not use GPU even if available")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # Proactive memory clearing
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
    
    # 1. Identify query indices in dataset
    target_ids = [s.strip() for s in args.query_ids.split(",")]
    query_indices = []
    for tid in target_ids:
        idx_list = dataset.df.index[dataset.df['sequence_id'] == tid].tolist()
        if idx_list:
            query_indices.append(idx_list[0])
        else:
            print(f"Warning: {tid} not found in test split.")
            
    if not query_indices:
        print("Error: No valid query IDs found.")
        return

    # 2. Prepare Gallery Subset (for comparison)
    # We take all unique IDs to have a meaningful search, but limited total count
    gallery_indices = list(range(len(dataset)))
    if len(gallery_indices) > args.gallery_subset:
        # Keep queries in gallery to ensure they could be found (though we'll exclude them)
        other_indices = [i for i in gallery_indices if i not in query_indices]
        np.random.seed(42)
        gallery_indices = query_indices + list(np.random.choice(other_indices, args.gallery_subset, replace=False))
    
    # Get num_classes from train split
    train_dataset = MARSDataset(cfg['data']['index_csv'], cfg['data']['root'], split='train')
    num_classes = train_dataset.num_classes

    # Model Setup
    if 'vit' in cfg['experiment']['name'].lower():
        model = ViTEmbedder(cfg['model']['backbone'], embed_dim=cfg['model']['embed_dim'], num_classes=num_classes, img_size=tuple(cfg['data']['image_size']))
    else:
        model = CNNEmbedder(cfg['model']['backbone'], embed_dim=cfg['model']['embed_dim'], num_classes=num_classes)
    
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    
    try:
        model = model.to(device)
        print(f"Model successfully loaded to {device}")
    except:
        print(f"Warning: GPU initialization failed. Falling back to CPU for visualization.")
        device = torch.device("cpu")
        model = model.to(device)
    
    # Ensure temporal_pool uses the same device (which might have changed to CPU)
    temporal_pool = TemporalPooling().to(device)
    
    # 3. Extract Features for Gallery Subset
    all_feats = extract_features_selective(model, temporal_pool, dataset, gallery_indices, device)
    all_pids = torch.tensor([dataset.df.iloc[i]['person_id'] for i in gallery_indices])
    
    # Mapping back gallery_indices to absolute local indices in 'all_feats'
    gallery_idx_map = {abs_idx: local_idx for local_idx, abs_idx in enumerate(gallery_indices)}

    # 4. Perform search for each query
    output_root = os.path.join("outputs", cfg['experiment']['name'], "visual_search")
    os.makedirs(output_root, exist_ok=True)

    for q_abs_idx in query_indices:
        q_local_idx = gallery_idx_map[q_abs_idx]
        q_feat = all_feats[q_local_idx].unsqueeze(0)
        q_pid = all_pids[q_local_idx].item()
        q_seq_id = dataset.df.iloc[q_abs_idx]['sequence_id']
        
        # Distances
        dist = torch.mm(q_feat, all_feats.t()).squeeze(0)
        indices = torch.argsort(dist, descending=True)
        
        # Exclude self
        top_indices = []
        for idx in indices:
            if idx.item() != q_local_idx:
                top_indices.append(idx.item())
            if len(top_indices) >= args.top_k: break
            
        # Plot
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(1, args.top_k + 1, 1)
        ax.imshow(get_representative_frame(dataset, q_abs_idx))
        ax.set_title(f"QUERY\n{q_seq_id}\nID: {q_pid}")
        ax.axis('off')
        
        for i, match_local_idx in enumerate(top_indices):
            ax = fig.add_subplot(1, args.top_k + 1, i + 2)
            match_abs_idx = gallery_indices[match_local_idx]
            match_pid = all_pids[match_local_idx].item()
            match_seq_id = dataset.df.iloc[match_abs_idx]['sequence_id']
            
            is_correct = (match_pid == q_pid)
            color = 'green' if is_correct else 'red'
            
            ax.imshow(get_representative_frame(dataset, match_abs_idx))
            # Show both PID and the specific Sequence/Cycle ID
            ax.set_title(f"Rank {i+1}\nPID: {match_pid}\n{match_seq_id}\nSim: {dist[match_local_idx]:.3f}", 
                         color=color, fontsize=10)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(output_root, f"search_{q_seq_id}.png"))
        print(f"Result saved for {q_seq_id}")
        plt.close()

if __name__ == "__main__":
    main()
