import torch
import argparse
from model.cnn_embedder import CNNEmbedder
from train_base import train_model, load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cnn_config.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from preprocess.dataset import MARSDataset
    dataset = MARSDataset(cfg['data']['index_csv'], cfg['data']['root'], split='train')
    
    model = CNNEmbedder(
        backbone_name=cfg['model']['backbone'],
        pretrained=cfg['model']['pretrained'],
        embed_dim=cfg['model']['embed_dim'],
        num_classes=dataset.num_classes,
        dropout=cfg['model']['dropout']
    )
    
    train_model(cfg, model, device)

if __name__ == "__main__":
    main()
