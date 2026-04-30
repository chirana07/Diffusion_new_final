# DAY 2 — Fine-tune from the Day 1 checkpoint (the master runbook)

**This is v2 of the runbook.** The original from-scratch plan was overfitting
(val PSNR stuck at 17 dB after 34 epochs while the loss was climbing). The new
plan **fine-tunes** the existing Day 1 `final.pth` (which already gets 22.77
PSNR on LOL-v2 Real) for 50 epochs at low learning rate. Two parallel runs,
with and without the illumination prior, give a clean method ablation.

Expected wall clock: **2 h of your active time** + **~3.5 h of Kaggle GPU**.

---

## What changed since the v1 runbook

If you ran v1 and ended up training from scratch, **stop that training now**.
Discard the `best_lolv2real_baseline.pth` and `best_lolv2real_illum.pth`
files if they exist — they're undertrained.

The new plan:

| | v1 (failed) | v2 (this plan) |
|---|---|---|
| Init | random | from Day 1 `final.pth` |
| Learning rate | 1e-4 | **1e-5** (10× lower) |
| Epochs | 120 | **50** |
| `w_ssim` | 0.5 | **0.3** (lower, prevents loss climbing) |
| Wall time per run | ~3 h | **~1.5 h** |
| Expected PSNR @ 5 steps | 17 dB | **23–24+ dB** |

The architectural change (illumination prior) is unchanged. What's changed
is the *training recipe*: we adapt a working model to LOL-v2 Real instead of
training a new one from random weights.

---

## Step 0 — Push your local commits (do this FIRST)

The notebooks need three new things on GitHub:
1. `kaggle_path_discovery.py` — robust dataset auto-discovery module
2. `train.py` with `--init-from`, `--lr`, `--dataset-root` flags
3. `kaggle_train_illum.ipynb` and `kaggle_eval_illum.ipynb` v2

On your laptop:

```bash
cd "/Users/chirana/Desktop/UOM/Sem 3/Software project IFS/Diffusion_new/LUMIDIFF/New_dif"
git add .
git commit -m "Day 2 v2: fine-tune from Day 1 ckpt, robust path discovery, smoke test"
git push
```

Open `https://github.com/chirana07/Diffusion_new_final` in a browser to
confirm the push went through. Cell 2 of each notebook explicitly checks the
new flags exist on GitHub and refuses to start if they don't.

---

## Step 1 — Confirm three datasets are on Kaggle

You'll attach all three to the training notebook in Step 2:

1. **LOL-v2** — must include both `Real_captured/Train/` and `Real_captured/Test/`. (The Train folder is the new requirement; Day 1 only used Test.)
2. **LOL eval15** — optional for training but useful for the eval notebook later.
3. **Day 1 checkpoint** — your existing `final.pth`. This is what we fine-tune from. It's probably already on Kaggle as a dataset (e.g. `lumidiff-checkpoint`).

If LOL-v2 on Kaggle is missing the Train folder, re-upload your local
`New_dif/LOL-v2/` folder as a **new** Kaggle dataset (don't try to update the
existing one — Kaggle's update flow is finicky).

---

## Step 2 — Run the training notebook

### 2.1 Open Kaggle

1. **kaggle.com/code → + New Notebook**
2. Right panel → **Settings → Accelerator → GPU T4 x1 → Save**
3. Right panel → **Persistence → Variables and Files**
4. Right panel → **+ Add Data** three times → attach LOL-v2, LOL eval15 (optional), and your Day 1 checkpoint dataset.

### 2.2 Import + run

Drag `kaggle_train_illum.ipynb` from your laptop onto the Kaggle page (or
**File → Import Notebook**), then click **Run All** (▶▶ at the top).

The 11 cells, in order:

| # | What it does | Wall time | Check |
|---|---|---|---|
| 1 | pip install | 30 s | last line says `Successfully installed ...` |
| 2 | clone repo + verify flags | 10 s | prints `All required flags present in train.py.` |
| 3 | **PREFLIGHT**: discover datasets | 5 s | prints all candidates + `Validated counts: ...` showing Train≥300 Test≥15 |
| 4 | locate Day 1 checkpoint | 1 s | prints `Will fine-tune from: ...` and the file size |
| 5 | GPU sanity | 1 s | `cuda: True ... Tesla T4` |
| 6 | **SMOKE TEST** (1 epoch, 5 val batches) | 2–3 min | runs to completion without error; prints `Smoke test passed.` |
| 7 | print fine-tune config | instant | shows `lr=1e-5, w_ssim=0.3, epochs=50` |
| 8 | **Run A — baseline_ft** (no prior) | ~1.5 h | per-epoch progress; val PSNR every 5 epochs should climb |
| 9 | **Run B — illum_ft** (with prior) | ~1.5 h | same |
| 10 | bundle checkpoints + zip | 30 s | writes `day2_checkpoints.zip` |

### 2.3 Sanity-checking during training (Cell 8 / Cell 9)

The per-epoch output shows e.g.
```
Epoch  3: 100%|██████████| 172/172 [01:24<00:00, ...]
[epoch 4] val PSNR 22.812  SSIM 0.7951  -> new best, saved best_baseline_ft.pth
```

What to expect for Run A (`baseline_ft`):
- Epoch 1 val PSNR: should be **close to 22.7 dB** (i.e. close to your Day 1 baseline). If it's <20 dB, the smart-load didn't work — stop and tell me.
- Epochs 5–50: val PSNR should climb gradually, ending in the **23.0–24.5 dB** range.

What to expect for Run B (`illum_ft`):
- Epoch 1 val PSNR: should be **close to 22.7 dB** because the illumination-prior input channel's weights are zero-initialized, so behavior matches the original at iteration 0.
- Epochs 5–50: should also climb. The +illum gain over baseline_ft is what we're measuring.

If val PSNR at epoch 5 is below 20 dB for either run, **stop the cell** and
inspect — the smart-load probably failed and we need to debug.

### 2.4 If the session disconnects mid-training

Kaggle saves `best_<tag>.pth` every time validation PSNR improves. So even a
mid-training disconnect leaves you with a usable checkpoint. Download
whatever's in `/kaggle/working/Diffunet/checkpoints/best_*.pth` via the
**Output** tab.

If you only get partway through the 50 epochs, that's still fine; the
checkpoint at epoch 25 with 1e-5 fine-tuning is likely already better than
what we started with.

### 2.5 Download and re-upload as a Kaggle dataset

After Cell 10 finishes:
1. Right panel → **Output tab** → download `day2_checkpoints.zip`.
2. Unzip locally — should contain `best_baseline_ft.pth`, `best_illum_ft.pth`,
   plus train logs.
3. **Datasets → + New Dataset → upload all the .pth files** → name it
   `lumidiff-day2-ckpts` → set Private → Create.

---

## Step 3 — Run the eval notebook

### 3.1 Open Kaggle

Same setup as Step 2 (GPU T4) but attach **four** datasets this time:
- LOL eval15
- LOL-v2
- Day 1 checkpoint (kept for reference; not strictly needed)
- `lumidiff-day2-ckpts` (the one you just uploaded)

### 3.2 Import + run

Drag `kaggle_eval_illum.ipynb` onto the page, **Run All**.

| # | What | Wall time |
|---|---|---|
| 1 | install | 30 s |
| 2 | clone repo | 10 s |
| 3 | preflight datasets | 5 s |
| 4 | locate checkpoints | 1 s |
| 5 | GPU sanity | 1 s |
| 6 | headline eval @ 5 + 20 steps with LPIPS | ~10 min |
| 7 | method ablation (baseline_ft @ 5) | ~5 min |
| 8 | step ablation (5/10/20/50/100) | ~30 min |
| 9 | ARR grid (alpha 0.0–0.5) | ~5 min |
| 10 | T4 efficiency benchmark | ~2 min |
| 11 | Render Figures 1 + 3 | 30 s |
| 12 | **Print Tables A/B/C/D** | instant — copy these into the paper |
| 13 | bundle + zip | 30 s |

### 3.3 Read the output of Cell 12 carefully

Expected results (numbers will vary based on training):

**Table A (headline, illum_ft):** PSNR on LOL-v2 Real should be 23.0–24.5 dB
at 5 steps (vs 22.77 from Day 1). LPIPS should be lower (better) than 0.214.

**Table B (method ablation):** the `+ illum prior (ours)` row should beat
`baseline_ft` by 0.1–0.5 dB. If it doesn't, that's still publishable as an
honest negative-result ablation — see fallback in `NOVELTY_METHOD.md`.

**Table C (step ablation):** PSNR should be roughly flat 5→100 steps,
matching the Day 1 finding. If it's not flat, your "5 steps is enough"
claim no longer holds for this checkpoint and we need to discuss.

**Table D (ARR):** best alpha gets a small bump (e.g. 0.1–0.3 dB) or
ties at alpha=0. Report whichever wins.

### 3.4 Download

Output tab → `phase3_day2_eval_outputs.zip` → unzip locally next to your
manuscript.

---

## What to do if something breaks

### Cell 2 says "missing flag in train.py"
You forgot Step 0 (push). On laptop: `git push`, then re-run Cell 2.

### Cell 3 prints "LOL-v2 Real TRAIN split not found"
The LOL-v2 dataset on Kaggle is missing the Train folder. Either:
- Re-upload LOL-v2 with both Train AND Test, or
- Manually override `discoveries['lolv2_real_train']` in Cell 3 (commented
  template provided).

### Cell 6 (smoke test) fails with shape mismatch
The `smart_load_checkpoint` shape-tolerance logic isn't matching your Day 1
checkpoint. Send me the exact error message — likely the model architecture
constants in `config.py` have drifted.

### Cell 8 epoch-1 val PSNR is < 20 dB
Smart-load probably didn't initialize the model from the checkpoint correctly.
Stop, inspect the smart_load log lines printed at startup of Cell 8 — should
say something like `loaded=N partial=1 skipped=0`. If `loaded=0`, the
checkpoint's state-dict keys don't match.

### Loss climbing again during Cell 8 or Cell 9
Lower `w_ssim` further: edit Cell 7, set `FT_WSSIM = "0.2"`, re-run. If still
unstable, lower the LR: `FT_LR = "5e-6"`.

### The +illum model is *worse* than baseline_ft
Possible. Two responses:
1. Report the negative-result ablation honestly. See `NOVELTY_METHOD.md`
   alternative paragraph.
2. Re-train illum_ft for 30 more epochs with the same settings. (The
   illumination prior may need more time to learn its 7th-channel weights.)

### LPIPS is empty in the eval table
`pip install lpips` in Cell 1 failed silently. Run `!pip install lpips` in
a fresh cell, then re-run eval Cells 6 + 12.

---

## Submission-day checklist (Day 4)

Before you click submit, every box must be ticked:

- [ ] Title doesn't contain "LuminaDiff" (or has a clearly different framing).
- [ ] Abstract pitches the **efficiency** story, not the absolute-quality story.
- [ ] Headline numbers in Table 1 are from the **fine-tuned illum_ft** checkpoint.
- [ ] LPIPS column filled in for every headline row.
- [ ] Latency reported in **ms/image on a single T4**.
- [ ] Figure 1 is the new render from `kaggle_eval_illum`, not Day 1.
- [ ] Figure 3 (step ablation) shows the few-step convergence holds for `illum_ft`.
- [ ] Method-ablation table (`baseline_ft` vs `illum_ft`) is in Section 4 with both rows.
- [ ] ARR ablation table somewhere; if best alpha is 0, report honestly.
- [ ] Recent diffusion-LLIE baselines cited (Diff-Retinex, PyDiff, LightenDiffusion).
- [ ] Reproducibility appendix lists the **exact fine-tuning command** (`--init-from`, `--lr 1e-5`, `--epochs 50`).
- [ ] Limitations section discusses LOL-v2 Synthetic + at least one failure case.
- [ ] Broader Impact paragraph mentions surveillance.
- [ ] Typo "It has been trained a U-net..." is fixed.
- [ ] GitHub repo has a `README.md` and is reachable from the paper's URL.
- [ ] All 14 boxes ticked.

---

## Reference: every file in the workspace

| File | Purpose |
|---|---|
| `DAY2_GO.md` (this) | Master runbook — follow top to bottom |
| `kaggle_path_discovery.py` | Robust dataset auto-discovery module |
| `kaggle_train_illum.ipynb` | Fine-tuning training notebook (smoke test + 2 runs) |
| `kaggle_eval_illum.ipynb` | Re-evaluation notebook |
| `train.py` | Updated with `--init-from`, `--lr`, `--dataset-root` |
| `evaluation.py` | Updated with `--gate-alpha`, `--gate-floor` |
| `diffusion.py` | Updated with Adaptive Residual Rescaling in `ddim_sample` |
| `REVIEWS.md` | All three reviews verbatim |
| `REVIEWER_RESPONSE.md` | 12 concerns mapped to actions |
| `NOVELTY_METHOD.md` | Section 3.X writeup for ARR + contribution bullets |
| `RESUBMISSION_PLAN.md`, `PHASE3_FEW_DAYS.md`, `RUNBOOK.md`, `KAGGLE_HOWTO.md` | Earlier plans, kept for reference |
