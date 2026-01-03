import torch
import argparse
from model.vit_embedder import ViTEmbedder
from train_base import train_model, load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vit_config.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We need num_classes from dataset to initialize model, 
    # but train_base handles it. To keep model initialization here:
    from preprocess.dataset import MARSDataset
    dataset = MARSDataset(cfg['data']['index_csv'], cfg['data']['root'], split='train')
    
    model = ViTEmbedder(
        backbone_name=cfg['model']['backbone'],
        pretrained=cfg['model']['pretrained'],
        embed_dim=cfg['model']['embed_dim'],
        num_classes=dataset.num_classes,
        dropout=cfg['model']['dropout'],
        img_size=tuple(cfg['data']['image_size'])
    )
    
    train_model(cfg, model, device)

if __name__ == "__main__":
    main()
