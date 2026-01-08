import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import json
import pandas as pd
import numpy as np

from preprocess.my_dataset import MyDataset
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
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--arch", type=str, choices=['vit', 'cnn'], required=True)
    parser.add_argument("--input-csv", type=str, default="inputs/sequences_labeled.csv")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_transforms = get_transforms(tuple(cfg['data']['image_size']), is_train=False)
    
    # Load dataset with metadata support
    test_dataset = MyDataset(
        index_csv=args.input_csv,
        root_dir=cfg['data']['root'],
        transform=test_transforms,
        sequence_len=cfg['data']['sequence_len'],
        split='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=cfg['data']['num_workers']
    )
    
    # Load Checkpoint first to get correct num_classes
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Infer num_classes from classifier weight shape
    if 'classifier.weight' in state_dict:
        num_classes = state_dict['classifier.weight'].shape[0]
        print(f"Inferred {num_classes} classes from checkpoint.")
    else:
        # Fallback to dataset if not found (unexpected for these models)
        full_dataset = MyDataset(args.input_csv, cfg['data']['root'], split=None)
        num_classes = full_dataset.num_classes
    
    if args.arch == 'vit':
        model = ViTEmbedder(
            backbone_name=cfg['model']['backbone'],
            pretrained=False,
            embed_dim=cfg['model']['embed_dim'],
            num_classes=num_classes,
            img_size=tuple(cfg['data']['image_size'])
        )
    else:
        model = CNNEmbedder(
            backbone_name=cfg['model']['backbone'],
            pretrained=False,
            embed_dim=cfg['model']['embed_dim'],
            num_classes=num_classes
        )
        
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    temporal_pool = TemporalPooling().to(device)
    temporal_pool.eval()
    
    all_features = []
    all_pids = []
    all_camids = []
    all_illum = []
    all_occl = []
    
    print(f"Extracting features for {len(test_dataset)} sequences...")
    with torch.no_grad():
        for frames, labels, metadata in tqdm(test_loader):
            frames = frames.to(device)
            B, T, C, H, W = frames.shape
            frames_flat = frames.view(B*T, C, H, W)
            
            feat_flat = model(frames_flat)
            embedding = temporal_pool(feat_flat.view(B, T, -1))
            
            if cfg.get('eval', {}).get('normalize_embeddings', True):
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
            all_features.append(embedding.cpu())
            all_pids.extend(labels.tolist())
            
            # Metadata Handling
            all_camids.extend(metadata['cam_id'])
            all_illum.extend(metadata['illum_label'])
            all_occl.extend(metadata['occlusion_label'])
    
    all_features = torch.cat(all_features, dim=0)
    
    # Process camera IDs to ints (e.g. 'c001' -> 1)
    camids = [int(str(c).replace('c', '')) if isinstance(c, str) and 'c' in c else int(c) for c in all_camids]
    pids = [test_dataset.pids[i] for i in all_pids]
    
    # Helper for categorized metrics
    def get_categorized_metrics(feat, pids, camids, labels, category_name):
        unique_cats = sorted(list(set(labels)))
        cat_results = {}
        
        # Whole set baseline
        print(f"Calculating baseline metrics...")
        mAP_all = calculate_map(feat, pids, camids, feat, pids, camids)
        cat_results['OVERALL'] = {'mAP': mAP_all, 'count': len(pids)}
        
        for cat in unique_cats:
            mask = np.array([l == cat for l in labels])
            if not np.any(mask): continue
            
            q_feat = feat[mask]
            q_pids = [pids[i] for i, m in enumerate(mask) if m]
            q_camids = [camids[i] for i, m in enumerate(mask) if m]
            
            print(f"Calculating metrics for {category_name}: {cat} ({len(q_pids)} sequences)...")
            # Query is categorized set, Gallery is the WHOLE test set
            mAP = calculate_map(q_feat, q_pids, q_camids, feat, pids, camids)
            cat_results[cat] = {'mAP': mAP, 'count': len(q_pids)}
            
        return cat_results

    print("\n--- Evaluation by Illumination ---")
    illum_metrics = get_categorized_metrics(all_features, pids, camids, all_illum, "Illumination")
    
    print("\n--- Evaluation by Occlusion ---")
    occl_metrics = get_categorized_metrics(all_features, pids, camids, all_occl, "Occlusion")
    
    final_results = {
        'arch': args.arch,
        'illumination': illum_metrics,
        'occlusion': occl_metrics
    }
    
    print("\nFinal Results Summary:")
    print(json.dumps(final_results, indent=4))
    
    output_dir = os.path.join("outputs", cfg['experiment']['name'])
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"categorized_eval_{args.arch}.json")
    with open(out_file, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
