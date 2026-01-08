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
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Extracting features"):
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
    parser.add_argument("--query-id", type=int, default=None, help="Specific Person ID to query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gallery-subset", type=int, default=1000, help="Limit gallery for memory safety")
    parser.add_argument("--force-cpu", action="store_true", help="Do not use GPU even if available")
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
    
    # Subset Gallery
    indices = list(range(len(dataset)))
    if len(indices) > args.gallery_subset:
        #np.random.seed(42)
        indices = list(np.random.choice(indices, args.gallery_subset, replace=False))
    
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
        print(f"Warning: GPU initialization failed. Falling back to CPU.")
        device = torch.device("cpu")
        model = model.to(device)
    
    temporal_pool = TemporalPooling().to(device)
    
    # Extract
    all_feats = extract_features_selective(model, temporal_pool, dataset, indices, device)
    all_pids = torch.tensor([dataset.df.iloc[i]['person_id'] for i in indices])
    
    # Pick a query from our subset
    if args.query_id is not None:
        q_local_indices = np.where(all_pids.numpy() == args.query_id)[0]
        if len(q_local_indices) == 0:
            print(f"ID {args.query_id} not found in the subset. Picking random.")
            q_local_idx = np.random.randint(len(all_feats))
        else:
            q_local_idx = q_local_indices[0]
    else:
        q_local_idx = np.random.randint(len(all_feats))

    q_abs_idx = indices[q_local_idx]
    q_feat = all_feats[q_local_idx].unsqueeze(0)
    q_pid = all_pids[q_local_idx].item()
    q_seq_id = dataset.df.iloc[q_abs_idx]['sequence_id']
    
    # Distances
    dist = torch.mm(q_feat, all_feats.t()).squeeze(0)
    sorted_indices = torch.argsort(dist, descending=True)
    
    # Exclude self
    top_indices = []
    for idx in sorted_indices:
        if idx.item() != q_local_idx:
            top_indices.append(idx.item())
        if len(top_indices) >= args.top_k: break
            
    # Plotting
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(1, args.top_k + 1, 1)
    ax.imshow(get_representative_frame(dataset, q_abs_idx))
    ax.set_title(f"QUERY\nPID: {q_pid}\n{q_seq_id}")
    ax.axis('off')
    
    for i, match_local_idx in enumerate(top_indices):
        ax = fig.add_subplot(1, args.top_k + 1, i + 2)
        match_abs_idx = indices[match_local_idx]
        match_pid = all_pids[match_local_idx].item()
        match_seq_id = dataset.df.iloc[match_abs_idx]['sequence_id']
        
        is_correct = (match_pid == q_pid)
        color = 'green' if is_correct else 'red'
        
        ax.imshow(get_representative_frame(dataset, match_abs_idx))
        ax.set_title(f"Rank {i+1}\nPID: {match_pid}\n{match_seq_id}\nSim: {dist[match_local_idx]:.3f}", 
                     color=color, fontsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f"Visual Re-ID Search - Model: {cfg['experiment']['name']}")
    plt.tight_layout()
    
    out_dir = os.path.join("outputs", cfg['experiment']['name'], "visual_search")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"random_search_pid{q_pid}.png")
    plt.savefig(out_file)
    print(f"Search results saved to {out_file}")
    plt.show()

if __name__ == "__main__":
    main()
