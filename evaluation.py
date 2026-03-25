import os
import glob
import torch
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
from skimage.measure import shannon_entropy
from skimage.filters import laplace

from config import Config
from model import ResidualConditionedUNet
from diffusion import DiffusionEngine

def calculate_metrics(img_pred, img_gt):
    # img_pred, img_gt are PIL images
    pred_np = np.array(img_pred)
    gt_np = np.array(img_gt)
    
    # Reference metrics
    p = psnr_func(gt_np, pred_np, data_range=255)
    s = ssim_func(gt_np, pred_np, data_range=255, channel_axis=2)
    
    # Non-reference metrics (on predicted image)
    # 1. Entropy (Information content)
    e = shannon_entropy(pred_np)
    
    # 2. Sharpness (Laplacian variance)
    gray_pred = np.array(img_pred.convert('L'))
    sharpness = laplace(gray_pred).var()
    
    return p, s, e, sharpness

def main():
    conf = Config()
    eval_dir = "./eval15"
    low_dir = os.path.join(eval_dir, "low")
    high_dir = os.path.join(eval_dir, "high")
    
    results_save_dir = "./eval_results"
    os.makedirs(results_save_dir, exist_ok=True)
    
    device = conf.DEVICE
    print(f"Using device: {device}")
    
    model = ResidualConditionedUNet().to(device)
    diff = DiffusionEngine()
    
    # Load checkpoint
    checkpoint_path = os.path.join(conf.SAVE_DIR, "last_pth_only", "final.pth")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(conf.SAVE_DIR, "final.pth")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    image_names = sorted(os.listdir(low_dir))
    metrics_list = []
    
    print(f"Evaluating on {len(image_names)} images in eval15...")
    
    for name in tqdm(image_names):
        start_time = time.time()
        low_path = os.path.join(low_dir, name)
        high_path = os.path.join(high_dir, name)
        
        low_pil = Image.open(low_path).convert("RGB")
        high_pil = Image.open(high_path).convert("RGB")
        
        # Resize to multiple of 8 (or fixed size as per config)
        # Using fixed size for standard evaluation
        low_resized = low_pil.resize((conf.IMG_SIZE, conf.IMG_SIZE))
        high_resized = high_pil.resize((conf.IMG_SIZE, conf.IMG_SIZE))
        
        low_tensor = (TF.to_tensor(low_resized) - 0.5) * 2.0
        low_tensor = low_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            gen_tensor = diff.sample(model, low_tensor)
            
        end_time = time.time()
        runtime = end_time - start_time
        
        gen_tensor = (gen_tensor + 1.0) / 2.0
        gen_tensor = torch.clamp(gen_tensor, 0.0, 1.0)
        
        gen_pil = TF.to_pil_image(gen_tensor.squeeze(0).cpu())
        
        # Calculate metrics
        p, s, e, sh = calculate_metrics(gen_pil, high_resized)
        metrics_list.append([p, s, e, sh, runtime])
        
        # Save output
        gen_pil.save(os.path.join(results_save_dir, f"eval_{name}"))
        
    metrics_arr = np.array(metrics_list)
    avg_metrics = metrics_arr.mean(axis=0)
    
    print("\n--- Evaluation Results ---")
    print(f"Avg PSNR: {avg_metrics[0]:.4f}")
    print(f"Avg SSIM: {avg_metrics[1]:.4f}")
    print(f"Avg Runtime per image: {avg_metrics[4]:.4f}s")
    print(f"Avg Entropy: {avg_metrics[2]:.4f} (bit)")
    print(f"Avg Sharpness (Laplace Var): {avg_metrics[3]:.6f}")
    
    # Save results to file
    with open("eval_metrics.txt", "w") as f:
        f.write(f"Inference on eval15\n")
        f.write(f"PSNR: {avg_metrics[0]:.4f}\n")
        f.write(f"SSIM: {avg_metrics[1]:.4f}\n")
        f.write(f"Runtime per image: {avg_metrics[4]:.4f}s\n")
        f.write(f"Entropy: {avg_metrics[2]:.4f}\n")
        f.write(f"Sharpness: {avg_metrics[3]:.6f}\n")

if __name__ == "__main__":
    main()
