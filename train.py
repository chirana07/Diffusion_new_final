# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from copy import deepcopy
from tqdm import tqdm
import os

from config import Config
from dataset import LOLDataset
from model import ResidualConditionedUNet
from diffusion import DiffusionEngine
from modules import CompositeEnhancementLoss


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


def train():
    conf = Config()

    train_loader = DataLoader(
        LOLDataset("train"),
        batch_size=conf.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = ResidualConditionedUNet().to(conf.DEVICE)
    diff = DiffusionEngine()
    ema = EMA(model)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=conf.LR_START,
        betas=conf.BETAS,
        weight_decay=conf.WEIGHT_DECAY
    )

    scheduler1 = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=conf.WARMUP_EPOCHS
    )
    scheduler2 = CosineAnnealingLR(
        optimizer,
        T_max=conf.EPOCHS - conf.WARMUP_EPOCHS,
        eta_min=conf.LR_MIN
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[conf.WARMUP_EPOCHS]
    )

    criterion = CompositeEnhancementLoss(
        w_char=1.0,
        w_ssim=0.5,
        w_perc=0.1,
        w_color=0.05,
        w_tv=0.02,
    ).to(conf.DEVICE)

    print(f"Parameters: {count_parameters(model):,}")
    print("Training residual-conditioned diffusion model...")

    for epoch in range(conf.EPOCHS):
        model.train()
        pbar = tqdm(train_loader)

        for low, high in pbar:
            low = low.to(conf.DEVICE)
            high = high.to(conf.DEVICE)

            target_residual = torch.clamp(high - low, -1.0, 1.0)

            t = torch.randint(0, conf.TIMESTEPS, (low.size(0),), device=conf.DEVICE)
            noisy_residual, _ = diff.q_sample(
                target_residual,
                t,
                offset_noise_strength=0.1
            )

            pred_img, pred_residual = model(
                noisy_residual,
                t,
                low,
                return_residual=True
            )

            loss, logs = criterion(
                pred_img,
                high,
                pred_residual=pred_residual
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), conf.GRAD_CLIP)
            optimizer.step()
            ema.update(model)

            pbar.set_description(
                f"Ep {epoch} | total {logs['total']:.4f} | "
                f"char {logs['char']:.4f} | ssim {logs['ssim']:.4f} | perc {logs['perc']:.4f}"
            )

        scheduler.step()

        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": ema.shadow
            }, os.path.join(conf.SAVE_DIR, "final.pth"))


if __name__ == "__main__":
    train()