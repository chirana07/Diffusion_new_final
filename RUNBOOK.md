# LuminaDiff-R — execution runbook

Last updated: 2026-04-24

This is the ordered set of commands to run. Most steps are **on Kaggle** (free T4, 12 GB). A few are local on your Mac. Each step is self-contained.

Every step in Phase 1 uses your **existing** checkpoint — no retraining. Phase 2 requires Kaggle GPU time. Phase 3 is polish.

---

## What changed in the code

Short summary of the files I modified today so you know what to expect:

| File | Change |
|---|---|
| `evaluation.py` | Full-resolution eval (no more 128×128 resize). Multi-split support. Optional LPIPS. CSV output. **This alone should bump your headline PSNR noticeably.** |
| `diffusion.py` | Added a proper `ddim_sample()`. Kept old `sample()` as `dpm_posterior` for the sampler ablation. |
| `dataset.py` | LOL-v2 Real + Synthetic support, random 256 crops, rot/flip augmentation, multiple-of-8 padding for validation. |
| `modules.py` | New `IlluminationPrior` module (max-channel + Gaussian blur). No trainable params. |
| `model.py` | `ResidualConditionedUNet` now has a `use_illum_prior` flag (default **False**, so your existing checkpoint still loads). When True, feeds the illum map as an extra channel. |
| `config.py` | Adds `CROP_SIZE`, `VAL_EVERY`, `USE_ILLUM_PRIOR`, `SEED`. Most values are env-var overridable. |
| `train.py` | Random-crop training, validation every N epochs on LOL eval15 with DDIM sampling, best-checkpoint saving, CSV logging, seeding. |
| `inference.py` | Auto-detects `use_illum_prior` from the checkpoint. |
| `measure_efficiency.py` | **New.** Params/FLOPs/latency/peak-memory table. Reviewer XCNR's #1 ask. |
| `ablate_steps.py` | **New.** Sweeps sampling steps across one or more splits and both samplers. |
| `ablate_losses.py` | **New.** Trains short variants removing one loss term at a time. |
| `ablate_method.py` | **New.** Trains baseline vs. +illumination-prior and compares. |

Everything is backward-compatible with your current checkpoint at `checkpoints/last_pth_only/final.pth`. Existing weights still load; new flags are off by default.

---

## Prerequisite: push these changes to your Kaggle git repo

Before any Kaggle run:

```bash
cd "/Users/chirana/Desktop/UOM/Sem 3/Software project IFS/Diffusion_new/LUMIDIFF/New_dif"
git add -A
git commit -m "LuminaDiff-R: full-res eval, DDIM sampler, illumination prior, ablations"
git push origin main
```

Your Kaggle notebooks clone from `https://github.com/chirana07/Diffusion_new_final.git`, so pushing here makes the new code available to every subsequent Kaggle run.

You'll also need to attach the LOL-v2 dataset to the Kaggle notebook. If you haven't yet, search the Kaggle datasets for "LOL-v2" (or upload your copy as a private dataset) and attach it to your notebook. The paths below assume a common layout; adjust if your dataset structure differs.

---

## Phase 1 — no retraining. Do this TODAY.

Goal: see what your existing trained weights actually score once the evaluation protocol is fixed.

### Step 1.1 — Re-evaluate at full resolution on LOL eval15 (your existing checkpoint)

```bash
# On Kaggle, inside the cloned repo directory:
python evaluation.py \
    --splits eval15:./eval15/low:./eval15/high \
    --inference-steps 20 \
    --sampler dpm_posterior \
    --tag legacy_sampler_fullres
```

**Expected:** PSNR will very likely jump meaningfully. The 17.12 number in your paper was computed on 128×128 resized images; here we keep the native LOL resolution (400×600) and pad to multiples of 8 like inference.py already did. Any gap between the two numbers is purely a measurement artifact.

### Step 1.2 — Same eval with the proper DDIM sampler

```bash
python evaluation.py \
    --splits eval15:./eval15/low:./eval15/high \
    --inference-steps 20 \
    --sampler ddim \
    --tag ddim_fullres
```

**Expected:** DDIM may give a small additional PSNR/SSIM change (sometimes +, sometimes −; the existing legacy sampler was already deterministic). Either way you now have a proper DDIM curve to ablate, which is what the paper claims to be using.

### Step 1.3 — Run on LOL-v2 (this is the "bigger dataset" reviewers demanded)

Edit the LOL-v2 paths to match where your Kaggle dataset mounts. Typical Kaggle paths look like `/kaggle/input/lol-v2/Real_captured/Test/Low` — verify with a `!ls /kaggle/input/` in a cell first.

```bash
python evaluation.py \
    --splits \
        eval15:./eval15/low:./eval15/high \
        lolv2_real:/kaggle/input/lol-v2/Real_captured/Test/Low:/kaggle/input/lol-v2/Real_captured/Test/Normal \
        lolv2_syn:/kaggle/input/lol-v2/Synthetic/Test/Low:/kaggle/input/lol-v2/Synthetic/Test/Normal \
    --inference-steps 20 \
    --sampler ddim \
    --tag all_splits_ddim_20
```

**Expected:** LOL-v2 Real and Synthetic are much larger test sets (100 images each) than eval15 (15 images). If the model generalises, numbers stay similar to eval15. If not, we learn something.

### Step 1.4 — Sampling-step ablation (gives you the efficiency/quality curve)

```bash
python ablate_steps.py \
    --splits eval15:./eval15/low:./eval15/high \
             lolv2_real:/kaggle/input/lol-v2/Real_captured/Test/Low:/kaggle/input/lol-v2/Real_captured/Test/Normal \
    --steps 5 10 20 50 100 \
    --samplers ddim dpm_posterior \
    --plot
```

**Expected output:** a `./eval_results/step_ablation/combined.csv` plus a PNG plot of PSNR vs. sampling steps. This is Figure 5 in the new paper.

### Step 1.5 — Efficiency measurements

On Kaggle (T4):

```bash
python measure_efficiency.py \
    --resolution 400 600 \
    --steps 5 10 20 50 100 \
    --sampler ddim \
    --device cuda
```

On your Mac (CPU baseline for the paper — deployment-style number):

```bash
cd "/Users/chirana/Desktop/UOM/Sem 3/Software project IFS/Diffusion_new/LUMIDIFF/New_dif"
pip install -q fvcore   # or thop; either works
python measure_efficiency.py \
    --resolution 400 600 \
    --steps 5 10 20 50 100 \
    --sampler ddim \
    --device cpu
```

Both runs produce a markdown table ready to paste into the paper and a CSV at `./eval_results/efficiency.csv`.

### Phase 1 decision gate

After Phase 1 you'll have:

- Headline PSNR/SSIM on eval15, LOL-v2 Real, LOL-v2 Synthetic
- The PSNR-vs-steps curve for both samplers
- A full efficiency table

Look at the numbers. If the PSNR on LOL-v2 Real at 20 steps DDIM is ~20 dB or higher, you're already competitive with EnlightenGAN / Zero-DCE / Retinex-Net and the paper is saveable **without retraining**. Skip to Phase 3.

If PSNR is still below 19 dB on LOL-v2 Real, go to Phase 2.

---

## Phase 2 — retrain (Kaggle GPU time required)

### Step 2.1 — Retrain the baseline at 256×256 crops on LOL-v2 Real

Why: your current checkpoint was trained at 128×128 full-image resize, which caps the model's ability to handle real-resolution images. Random 256 crops on LOL-v2 will usually outperform it by a couple of dB.

```bash
# On Kaggle T4. 120 epochs at batch 4 / crop 256 fits comfortably in one session.
python train.py \
    --layout lolv2_real \
    --epochs 120 \
    --crop-size 256 \
    --batch-size 4 \
    --use-illum-prior 0 \
    --tag baseline_lolv2real
```

At the end you'll have `./checkpoints/best_baseline_lolv2real.pth`. Evaluate it:

```bash
python evaluation.py \
    --splits \
        eval15:./eval15/low:./eval15/high \
        lolv2_real:/kaggle/input/lol-v2/Real_captured/Test/Low:/kaggle/input/lol-v2/Real_captured/Test/Normal \
    --checkpoint ./checkpoints/best_baseline_lolv2real.pth \
    --inference-steps 20 --sampler ddim \
    --tag baseline_lolv2real_eval
```

### Step 2.2 — Train with the illumination prior (LuminaDiff-R proper)

```bash
python train.py \
    --layout lolv2_real \
    --epochs 120 \
    --crop-size 256 \
    --batch-size 4 \
    --use-illum-prior 1 \
    --tag luminadiffR_lolv2real
```

Evaluate:

```bash
python evaluation.py \
    --splits \
        eval15:./eval15/low:./eval15/high \
        lolv2_real:/kaggle/input/lol-v2/Real_captured/Test/Low:/kaggle/input/lol-v2/Real_captured/Test/Normal \
        lolv2_syn:/kaggle/input/lol-v2/Synthetic/Test/Low:/kaggle/input/lol-v2/Synthetic/Test/Normal \
    --checkpoint ./checkpoints/best_luminadiffR_lolv2real.pth \
    --inference-steps 20 --sampler ddim \
    --tag luminadiffR_eval
```

### Step 2.3 — Method ablation (baseline vs +illum prior)

If you already did 2.1 and 2.2, you can run the ablation in eval-only mode:

```bash
python ablate_method.py \
    --layout lolv2_real \
    --splits lolv2_real:/kaggle/input/lol-v2/Real_captured/Test/Low:/kaggle/input/lol-v2/Real_captured/Test/Normal \
    --skip-train
```

It reads `./checkpoints/best_method_baseline.pth` and `./checkpoints/best_method_illum.pth`, so rename your 2.1 / 2.2 outputs accordingly, or just call the driver fresh with your target epoch budget.

### Step 2.4 — Loss-component ablation

Warning: this trains **7 models** from scratch. At 80 epochs each on T4 this is ~6-8 Kaggle sessions. Run it only if Phase 1's main numbers are already where you want them.

```bash
python ablate_losses.py \
    --layout lolv2_real \
    --epochs 80 \
    --crop-size 256 \
    --splits lolv2_real:/kaggle/input/lol-v2/Real_captured/Test/Low:/kaggle/input/lol-v2/Real_captured/Test/Normal
```

A cheaper alternative for a workshop-tier paper: train only the ones you're most curious about (e.g. `--variants full char_only no_perc no_tv`), which covers the interesting endpoints in 4 runs.

---

## Phase 3 — experiments done, paper writing

Things to do once the numbers are frozen:

1. **Collect the comparison table.** Put your new numbers next to Retinex-Net, URetinex-Net, Zero-DCE, EnlightenGAN, MSATr, and **at least two recent diffusion LLIE methods** (Diff-Retinex, PyDiff, LightenDiffusion, or GSAD — take their numbers from their original papers and mark them with an asterisk).
2. **Qualitative figure.** Pick 4-6 representative images (one very dark, one with heavy colour shift, one noisy, one already-OK-but-needs-tone). Place columns in this order: input | Zero-DCE | Retinex-Net | URetinex-Net | LuminaDiff-R | GT.
3. **Failure case figure.** Reviewers reward honesty. Include 2 images where the model struggles — e.g. saturated whites or very fine text — and discuss in Limitations.
4. **Rewrite the abstract and intro** around the Retinex-conditioned residual diffusion story. The old "multi-loss training" framing doesn't carry a novelty claim.
5. **Grammar pass.** The "It has been trained a U-net style denoiser" sentence must go. A single read-through in a grammar checker fixes most of the issues reviewers flagged.
6. **Add a reproducibility appendix** with exact hyperparameters, seed (42), Kaggle notebook URL, and a link to the GitHub repo.

---

## Troubleshooting

**"Could not resolve lolv2_real/train under root=..."**: the layout detector in `dataset.py` looks for `Real_captured/Train/Low` and `Real_captured/Train/Normal` under the root. If your LOL-v2 dataset uses different folder names (e.g. `Low` vs. `low` case, or `GT` instead of `Normal`), either rename the folders or edit the `LAYOUTS` dict in `dataset.py`.

**"fvcore failed"**: the efficiency script falls back to `thop` then to params-only reporting. Install one: `pip install fvcore` or `pip install thop`.

**"LPIPS not installed"**: `pip install lpips`. If you don't have time, leave it — it's optional. PSNR/SSIM is enough for the main table.

**Backward compat**: if you ever load a prior-enabled checkpoint but forget to set `use_illum_prior`, you'll get a shape mismatch on `head.weight`. `evaluation.py` and `inference.py` now auto-detect this from the saved checkpoint.

**"Workspace still starting" or Kaggle session OOM**: drop the batch size to 2 and the crop size to 192 and retry. The model is small enough that this still produces good numbers.

---

## What you need to do, in order

1. **Commit and push** the code changes (five minutes).
2. Run **Phase 1 steps 1.1 through 1.5** on Kaggle with your existing checkpoint. That's ~1 hour of T4 time total. Report back the numbers you get.
3. Based on those numbers, I'll tell you whether Phase 2 retraining is needed, or whether we can go straight to paper rewriting.

Don't start Phase 2 until we look at Phase 1 results together — there's a real chance the evaluation fix alone gets you competitive, and we save 2-3 days of Kaggle time.
