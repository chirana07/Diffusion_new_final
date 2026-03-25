import math
import torch
from config import Config


def cosine_beta_schedule(timesteps, s=0.008, max_beta=0.999):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-6, max_beta)


class DiffusionEngine:
    def __init__(self):
        self.conf = Config()
        self.device = self.conf.DEVICE
        self.steps = self.conf.TIMESTEPS

        self.betas = cosine_beta_schedule(self.steps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=self.device),
            self.alphas_cumprod[:-1]
        ], dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def make_noise(self, x, offset_noise_strength=0.05):
        noise = torch.randn_like(x)
        if offset_noise_strength > 0:
            offset = torch.randn(x.size(0), x.size(1), 1, 1, device=x.device)
            noise = noise + offset_noise_strength * offset
        return noise

    def q_sample(self, x_start, t, noise=None, offset_noise_strength=0.05):
        if noise is None:
            noise = self.make_noise(x_start, offset_noise_strength)

        s1 = self.sqrt_alphas_cumprod[t][:, None, None, None]
        s2 = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        x_t = s1 * x_start + s2 * noise
        return x_t, noise

    @torch.no_grad()
    def sample(self, model, low_light, inference_steps=None):
        model.eval()
        b = low_light.shape[0]

        inference_steps = inference_steps or getattr(self.conf, "INFERENCE_STEPS", self.steps)
        inference_steps = min(inference_steps, self.steps)

        residual = torch.randn_like(low_light)

        # evenly spaced deterministic schedule
        schedule = torch.linspace(
            self.steps - 1, 0, inference_steps, device=self.device
        ).long()
        schedule = torch.unique_consecutive(schedule)

        for i in schedule:
            t = torch.full((b,), int(i.item()), device=self.device, dtype=torch.long)

            pred_img, pred_residual = model(residual, t, low_light, return_residual=True)
            pred_residual = torch.clamp(pred_residual, -1.0, 1.0)

            posterior_mean = (
                self.posterior_mean_coef1[t][:, None, None, None] * pred_residual +
                self.posterior_mean_coef2[t][:, None, None, None] * residual
            )

            residual = posterior_mean

        final_img = torch.clamp(low_light + residual, -1.0, 1.0)
        return final_img