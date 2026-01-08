import os
import argparse
import yaml
import torch
import numpy as np
from PIL import Image
import glob
from preprocess.transforms import get_transforms
from model.vit_embedder import ViTEmbedder
from model.temporal_pooling import TemporalPooling

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Inference Visual Feature Extractor")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory containing frames of a sequence")
    parser.add_argument("--output", type=str, default="embedding.npy", help="Output file for embedding")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    print("Loading model...")
    # Note: we don't need num_classes for inference, but the model init might require it.
    # We can pass a dummy value or modify model to handle it.
    # Let's assume 0 or handle it in model.
    model = ViTEmbedder(
        backbone_name=cfg['model']['backbone'],
        pretrained=False, # No need to download pretrained weights again
        embed_dim=cfg['model']['embed_dim'],
        num_classes=0, # Dummy
        dropout=0.0
    ).to(device)
    
    temporal_pool = TemporalPooling().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # We need to filter out classifier weights if num_classes doesn't match or strictly load
    # Since we initialized with num_classes=0, the classifier weights won't match if trained with classes.
    # So we load with strict=False or filter.
    state_dict = checkpoint['model_state_dict']
    # Remove classifier weights from state_dict
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    temporal_pool.eval()
    
    # Prepare data
    transform = get_transforms(tuple(cfg['data']['image_size']), is_train=False)
    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")))
    
    if not frame_paths:
        print(f"No frames found in {args.frames_dir}")
        return
        
    frames = []
    for p in frame_paths:
        img = Image.open(p).convert('RGB')
        img = transform(img)
        frames.append(img)
        
    frames = torch.stack(frames).unsqueeze(0).to(device) # [1, T, C, H, W]
    
    # Inference
    with torch.no_grad():
        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B*T, C, H, W)
        features_flat = model(frames_flat) # [B*T, d]
        features = features_flat.view(B, T, -1)
        embedding = temporal_pool(features) # [B, d]
        
    # Save
    emb_np = embedding.cpu().numpy().squeeze()
    np.save(args.output, emb_np)
    print(f"Embedding saved to {args.output}")

if __name__ == "__main__":
    main()
