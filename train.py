"""
train.py — training driver for LuminaDiff / LuminaDiff-R.

Changes vs. prior version (2026-04-24):
- Seeded RNG (torch, numpy, python) for reproducibility.
- Random 256-crop training via the new dataset loader.
- Validation loop every VAL_EVERY epochs on LOL eval15 (full-resolution DDIM sampling)
  that tracks PSNR and saves best.pth whenever validation PSNR improves.
- Saves last.pth each epoch (overwrite) plus best.pth (best val PSNR).
- CSV logging of per-epoch train loss components and val PSNR/SSIM.
- Respects Config.USE_ILLUM_PRIOR.
- CLI args let you override dataset layout / epochs / prior without editing config.

Usage on Kaggle (T4):
    python train.py --layout lolv2_real --epochs 200 --crop-size 256
    python train.py --layout lolv2_real --use-illum-prior 1 --epochs 200
"""
import argparse
import csv
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import LOLDataset
from diffusion import DiffusionEngine
from model import ResidualConditionedUNet
from modules import CompositeEnhancementLoss


# --------------------------- utilities ---------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = deepcopy(model.state_dict())

    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = (
                        (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------- validation ---------------------------

@torch.no_grad()
def validate(model, diff, val_loader, device, max_batches=None):
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn

    model.eval()
    psnrs, ssims = [], []
    for i, (low, high) in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        low = low.to(device)
        high = high.to(device)
        pred = diff.ddim_sample(model, low, inference_steps=Config.INFERENCE_STEPS)
        # to [0,1]
        pred = torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)
        target = torch.clamp((high + 1.0) / 2.0, 0.0, 1.0)
        for b in range(pred.shape[0]):
            p_np = pred[b].cpu().permute(1, 2, 0).numpy()
            t_np = target[b].cpu().permute(1, 2, 0).numpy()
            psnrs.append(psnr_fn(t_np, p_np, data_range=1.0))
            ssims.append(ssim_fn(t_np, p_np, data_range=1.0, channel_axis=2))
    model.train()
    return float(np.mean(psnrs)) if psnrs else 0.0, float(np.mean(ssims)) if ssims else 0.0


# --------------------------- main ---------------------------

def train(args):
    conf = Config()
    set_seed(args.seed or conf.SEED)

    # Apply CLI overrides to env-driven config
    if args.crop_size:
        conf.CROP_SIZE = args.crop_size
    if args.epochs:
        conf.EPOCHS = args.epochs
    if args.batch_size:
        conf.BATCH_SIZE = args.batch_size
    if args.use_illum_prior is not None:
        conf.USE_ILLUM_PRIOR = bool(args.use_illum_prior)

    print(f"Config: crop={conf.CROP_SIZE}, batch={conf.BATCH_SIZE}, epochs={conf.EPOCHS}, "
          f"illum_prior={conf.USE_ILLUM_PRIOR}, layout={args.layout}")

    train_ds = LOLDataset(mode="train", layout=args.layout,
                          root=args.dataset_root,
                          crop_size=conf.CROP_SIZE, augment=True)
    val_ds = LOLDataset(mode="test", layout=args.layout,
                        root=args.dataset_root, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=conf.BATCH_SIZE, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    # Validation uses batch 1 because full-resolution images vary in size (padded to /8).
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = ResidualConditionedUNet(use_illum_prior=conf.USE_ILLUM_PRIOR).to(conf.DEVICE)
    diff = DiffusionEngine()
    ema = EMA(model)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=conf.LR_START,
        betas=conf.BETAS,
        weight_decay=conf.WEIGHT_DECAY,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=conf.WARMUP_EPOCHS),
            CosineAnnealingLR(optimizer, T_max=conf.EPOCHS - conf.WARMUP_EPOCHS, eta_min=conf.LR_MIN),
        ],
        milestones=[conf.WARMUP_EPOCHS],
    )

    criterion = CompositeEnhancementLoss(
        w_char=args.w_char, w_ssim=args.w_ssim, w_perc=args.w_perc,
        w_color=args.w_color, w_grad=args.w_grad, w_tv=args.w_tv,
    ).to(conf.DEVICE)

    print(f"Trainable params: {count_parameters(model):,}")

    # CSV logger
    os.makedirs(conf.SAVE_DIR, exist_ok=True)
    log_path = os.path.join(conf.SAVE_DIR, f"train_log_{args.tag or 'default'}.csv")
    log_f = open(log_path, "w", newline="")
    log_writer = csv.writer(log_f)
    log_writer.writerow([
        "epoch", "lr", "train_total", "char", "ssim", "perc", "color", "grad", "tv",
        "val_psnr", "val_ssim", "sec",
    ])

    best_psnr = -1.0
    for epoch in range(conf.EPOCHS):
        model.train()
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        running = {"total": 0, "char": 0, "ssim": 0, "perc": 0, "color": 0, "grad": 0, "tv": 0}
        n_batches = 0

        for low, high in pbar:
            low = low.to(conf.DEVICE, non_blocking=True)
            high = high.to(conf.DEVICE, non_blocking=True)

            target_residual = torch.clamp(high - low, -1.0, 1.0)

            t = torch.randint(0, conf.TIMESTEPS, (low.size(0),), device=conf.DEVICE)
            noisy_residual, _ = diff.q_sample(
                target_residual, t, offset_noise_strength=0.1
            )

            pred_img, pred_residual = model(noisy_residual, t, low, return_residual=True)
            loss, logs = criterion(pred_img, high, pred_residual=pred_residual)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), conf.GRAD_CLIP)
            optimizer.step()
            ema.update(model)

            for k in running:
                running[k] += float(logs[k])
            n_batches += 1

            pbar.set_postfix({
                "total": f"{logs['total']:.3f}",
                "char": f"{logs['char']:.3f}",
                "ssim": f"{logs['ssim']:.3f}",
            })

        scheduler.step()
        avg = {k: v / max(n_batches, 1) for k, v in running.items()}

        # Validation
        val_psnr, val_ssim = -1.0, -1.0
        if (epoch + 1) % conf.VAL_EVERY == 0 or epoch == conf.EPOCHS - 1:
            # load EMA weights into a shadow model for validation
            ema_model = ResidualConditionedUNet(use_illum_prior=conf.USE_ILLUM_PRIOR).to(conf.DEVICE)
            ema_model.load_state_dict(ema.shadow)
            val_psnr, val_ssim = validate(ema_model, diff, val_loader, conf.DEVICE,
                                          max_batches=args.val_max_batches)
            print(f"[epoch {epoch}] val PSNR {val_psnr:.3f}  SSIM {val_ssim:.4f}")
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "ema": ema.shadow,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "use_illum_prior": conf.USE_ILLUM_PRIOR,
                }, os.path.join(conf.SAVE_DIR, f"best_{args.tag or 'default'}.pth"))
                print(f"  -> new best, saved best_{args.tag or 'default'}.pth")

        # Save last
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.shadow,
            "use_illum_prior": conf.USE_ILLUM_PRIOR,
        }, os.path.join(conf.SAVE_DIR, f"last_{args.tag or 'default'}.pth"))

        log_writer.writerow([
            epoch, optimizer.param_groups[0]["lr"],
            avg["total"], avg["char"], avg["ssim"], avg["perc"],
            avg["color"], avg["grad"], avg["tv"],
            val_psnr, val_ssim, time.time() - t0,
        ])
        log_f.flush()

    log_f.close()
    print(f"Done. Best val PSNR: {best_psnr:.3f}")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default=os.environ.get("DATASET_LAYOUT", "lol_v1"),
                   help="Dataset layout: lol_v1 | lolv2_real | lolv2_syn | flat")
    p.add_argument("--dataset-root", default=None,
                   help="Path to the directory that CONTAINS the layout's relative paths "
                        "(e.g. for lolv2_real: the dir containing 'Real_captured/'). "
                        "Defaults to Config.DATASET_ROOT with glob fallback.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--crop-size", type=int, default=None)
    p.add_argument("--use-illum-prior", type=int, default=None, choices=[0, 1])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--tag", default=os.environ.get("RUN_TAG", "default"))
    p.add_argument("--val-max-batches", type=int, default=None,
                   help="Limit validation to N batches (for quick smoke tests)")
    # Loss weights — used by the loss-component ablation
    p.add_argument("--w-char", type=float, default=1.0)
    p.add_argument("--w-ssim", type=float, default=0.5)
    p.add_argument("--w-perc", type=float, default=0.1)
    p.add_argument("--w-color", type=float, default=0.05)
    p.add_argument("--w-grad", type=float, default=0.2)
    p.add_argument("--w-tv", type=float, default=0.02)
    return p.parse_args()


if __name__ == "__main__":
    train(parse())
