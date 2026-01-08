import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison():
    vit_dir = "outputs/vit_experiment"
    cnn_dir = "outputs/cnn_experiment"
    
    vit_history_path = os.path.join(vit_dir, "history.json")
    cnn_history_path = os.path.join(cnn_dir, "history.json")
    vit_eval_path = os.path.join(vit_dir, "eval_results.json")
    cnn_eval_path = os.path.join(cnn_dir, "eval_results.json")
    
    if not (os.path.exists(vit_history_path) and os.path.exists(cnn_history_path)):
        print("History files not found. Run training first.")
        return

    with open(vit_history_path, 'r') as f:
        vit_h = json.load(f)
    with open(cnn_history_path, 'r') as f:
        cnn_h = json.load(f)
        
    # Plot Training History
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(vit_h['train_loss'], label='ViT Train Loss', color='blue')
    plt.plot(cnn_h['train_loss'], label='CNN Train Loss', color='red')
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(vit_h['train_acc'], label='ViT Train Acc', color='blue', linestyle='--')
    plt.plot(cnn_h['train_acc'], label='CNN Train Acc', color='red', linestyle='--')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("outputs/history_comparison.png")
    plt.close()
    
    # Plot Evaluation Metrics
    if os.path.exists(vit_eval_path) and os.path.exists(cnn_eval_path):
        with open(vit_eval_path, 'r') as f:
            vit_e = json.load(f)
        with open(cnn_eval_path, 'r') as f:
            cnn_e = json.load(f)
            
        metrics = ['mAP', 'ratio_inter_intra']
        vit_values = [vit_e[m] for m in metrics]
        cnn_values = [cnn_e[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(8, 6))
        plt.bar(x - width/2, vit_values, width, label='ViT', color='skyblue')
        plt.bar(x + width/2, cnn_values, width, label='CNN', color='salmon')
        
        plt.ylabel('Score')
        plt.title('Final Evaluation Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        
        plt.savefig("outputs/metrics_comparison.png")
        plt.close()
        print("Comparison plots saved to outputs/")

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    plot_comparison()
