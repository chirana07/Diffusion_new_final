# sanity_test.py
import torch
from config import Config
from model import ResidualConditionedUNet
from diffusion import DiffusionEngine

conf = Config()
device = conf.DEVICE

model = ResidualConditionedUNet().to(device)
diff = DiffusionEngine()

low = torch.randn(2, 3, conf.IMG_SIZE, conf.IMG_SIZE, device=device).clamp(-1, 1)
high = torch.randn(2, 3, conf.IMG_SIZE, conf.IMG_SIZE, device=device).clamp(-1, 1)

target_residual = torch.clamp(high - low, -1.0, 1.0)
t = torch.randint(0, conf.TIMESTEPS, (2,), device=device)

noisy_residual, _ = diff.q_sample(target_residual, t)
pred_img, pred_residual = model(noisy_residual, t, low, return_residual=True)

print("low:", low.shape)
print("target_residual:", target_residual.shape)
print("noisy_residual:", noisy_residual.shape)
print("pred_img:", pred_img.shape)
print("pred_residual:", pred_residual.shape)