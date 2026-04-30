# DAY 2 — Train + Re-evaluate (the master runbook)

Follow this top to bottom. **Don't skip steps.** Every step has a check that
tells you if it worked. If a step fails, stop and fix before moving on.

Estimated wall clock: **2 hours of your active time**, plus **5–7 hours of
unattended Kaggle GPU time** spread over 1–2 sessions.

---

## What you'll have when this is done

- `best_lolv2real_illum.pth` — new checkpoint trained on LOL-v2 Real with the
  Retinex illumination prior enabled. **This is the new headline model.**
- (optional) `best_lolv2real_baseline.pth` — same training data, prior disabled,
  for the method-ablation row.
- New headline numbers (PSNR/SSIM/LPIPS at 5 and 20 DDIM steps).
- Method ablation table (baseline vs +illum), step ablation, ARR ablation.
- Updated Figure 1 (teaser) + Figure 3 (step ablation) PDFs.
- A T4 efficiency CSV for the new model.
- Everything bundled in `/kaggle/working/phase3_day2_eval_outputs.zip`.

---

## Step 0 — Push the latest commits to GitHub (do this FIRST)

Both Kaggle notebooks `git clone` your repo. They check that the new flags
(`--dataset-root` in `train.py`, `--gate-alpha` in `evaluation.py`) exist on
GitHub before running, and refuse to start if they don't.

On your laptop, in `New_dif/`:

```bash
git status
# you should see modified: train.py evaluation.py diffusion.py
# and untracked: kaggle_train_illum.ipynb kaggle_eval_illum.ipynb
#                kaggle_day2_novelty.ipynb DAY2_GO.md
#                NOVELTY_METHOD.md REVIEWS.md REVIEWER_RESPONSE.md ...

git add .
git commit -m "Day 2: --dataset-root, ARR (--gate-alpha), training + eval notebooks"
git push
```

**Verify push went through**: open
`https://github.com/chirana07/Diffusion_new_final` in a browser and confirm
the latest commit message is the one you just pushed.

---

## Step 1 — Make sure the LOL-v2 dataset is on Kaggle

You probably already have it from Day 1. If not: on Kaggle, go to **Datasets →
+ New Dataset**, upload your local `New_dif/LOL-v2/` folder (the one
containing `Real_captured/Train/Low` etc.), and call it `lol-v2`. Wait until
it shows up under your **Your Datasets** list.

You also need the eval15 dataset (already attached for Day 1) — same procedure
if you haven't uploaded it yet.

---

## Step 2 — Run the training notebook (the long one)

This is the part that will take ~5–7 hours. Plan it around your day. You can
walk away while it runs.

### 2.1 Open Kaggle and create a new notebook

1. Go to **kaggle.com/code → + New Notebook**.
2. Right panel → **Settings → Accelerator → GPU T4 x1** → **Save**.
3. Right panel → **Persistence → Variables and Files** (so the checkpoint
   survives if the session disconnects).
4. Right panel → **+ Add Data** → attach your LOL-v2 dataset. (No checkpoint
   dataset needed for training.)

### 2.2 Import the notebook

Drag-drop `New_dif/kaggle_train_illum.ipynb` from your laptop onto the Kaggle
page (or **File → Import Notebook**).

### 2.3 Run

Click **Run All** (▶▶ at the top). The cells in order:

| Cell | What it does | Check |
|---|---|---|
| 1 | pip install | last line says `Successfully installed ...` |
| 2 | git clone repo | prints `--dataset-root flag present in train.py: True` |
| 3 | discover LOL-v2 path | prints `DATASET_ROOT = /kaggle/input/.../LOL-v2` and shows ~689/689/100/100 file counts |
| 4 | GPU sanity | prints `cuda: True ... Tesla T4` |
| 5 | hyperparameters | prints config (epochs=120 etc.) |
| 6 | **train baseline** (optional) | runs ~2.5–3.5 hours; per-epoch progress bars |
| 7 | **train +illum** (REQUIRED) | runs ~2.5–3.5 hours; per-epoch progress bars |
| 8 | collect checkpoints + zip | writes `phase3_day2_outputs.zip` (different name: `day2_checkpoints.zip`) |

**Sanity-check during training** (in the per-epoch output of Cell 7): the
`val_psnr` line printed every 5 epochs should go up over time. Roughly:
- epoch 5: 16–18 dB (still warming up)
- epoch 50: 21–23 dB
- epoch 120: 23–25 dB on LOL eval15 *or* equivalent on the val set

If you see val_psnr stuck below 18 dB after epoch 30, something is wrong (stop
and ping me — likely a dataset-path issue, the training is using the wrong
folder).

**If you only have time for one run:** skip Cell 6, run Cell 7. The `+illum`
checkpoint is the one you actually need.

### 2.4 If Kaggle disconnects

Kaggle free tier kills sessions after ~12 hours of inactivity or if your
browser tab is closed too long. The good news: `train.py` saves
`best_lolv2real_illum.pth` *every time validation PSNR improves*, so even a
mid-training disconnect leaves you with a usable (if undertrained)
checkpoint. Just download whatever's in `/kaggle/working/Diffunet/checkpoints/`
via the Output tab.

If you need to resume: re-running Cell 7 starts from scratch (no resume
support). Better to let the partial best_*.pth stand and move to Step 3.

### 2.5 Download the checkpoints

When Cell 8 finishes:

1. Right panel → **Output tab** → find `day2_checkpoints.zip` → click **download**.
2. Unzip locally. You should see `best_lolv2real_illum.pth` (and optionally
   `best_lolv2real_baseline.pth`) plus `train_log_*.csv` files.

---

## Step 3 — Upload the new checkpoints as a Kaggle dataset

The eval notebook (Step 4) reads checkpoints from `/kaggle/input/`, not from
the working directory of a previous notebook. So you need to upload the
checkpoints once.

1. Kaggle → **Datasets → + New Dataset**.
2. Upload `best_lolv2real_illum.pth` (and `best_lolv2real_baseline.pth` if you
   have it) plus the `train_log_*.csv` files.
3. Name it `lumidiff-day2-ckpts`.
4. Set visibility to **Private**.
5. Click **Create**. Wait until it processes (~1–2 min for ~70 MB).

---

## Step 4 — Run the re-evaluation notebook

### 4.1 Open a new Kaggle notebook

Same setup as Step 2 (GPU T4 x1, persistence on), with **three** datasets
attached this time:
- LOL eval15
- LOL-v2
- `lumidiff-day2-ckpts` (the one you just uploaded)

### 4.2 Import + run

Drag-drop `kaggle_eval_illum.ipynb` → **Run All**.

| Cell | What it does | Check |
|---|---|---|
| 1 | pip install | `Successfully installed ...` |
| 2 | clone repo | prints cwd |
| 3 | discover datasets + checkpoints | prints all six paths; `ILLUM_CKPT` must be set |
| 4 | GPU sanity | `cuda: True` |
| 5 | headline eval (5 + 20 steps) | one block per (split × steps), tqdm progress bars |
| 6 | method ablation (baseline @ 5) | runs only if BASELINE_CKPT was found |
| 7 | step ablation (5/10/20/50/100) | five eval runs, ~30 min total |
| 8 | ARR grid (alpha 0–0.5) | six fast runs (5 min total), prints best alpha |
| 9 | T4 efficiency | quick CSV |
| 10 | render Figure 1 + Figure 3 | writes PDFs |
| 11 | **prints the four paper tables** | this is what you copy into the manuscript |
| 12 | bundle + zip | `phase3_day2_eval_outputs.zip` ready to download |

### 4.3 Download outputs

Output tab → `phase3_day2_eval_outputs.zip` → download. Unzip locally next
to the manuscript.

### 4.4 Sanity-check the numbers (in Cell 11 output)

Read the four printed tables. The signals you want to see:

- **Table A (headline +illum)** — PSNR on LOL-v2 Real should be *higher* than
  Day 1's 22.77 if the training helped. Realistic range: 23.0–25.0 dB.
- **Table B (method ablation)** — `+illum prior (ours)` row should beat
  `baseline (no prior)` on LOL-v2 Real. If it doesn't (gap < 0.2 dB), the
  illum prior didn't help on this dataset; that's fine for the paper, you'll
  report it as an honest negative result.
- **Table C (step ablation)** — PSNR should be ~flat from 5 to 100 steps,
  same shape as Day 1's curve. If 5-step lags 20-step by >0.5 dB, the new
  model lost the few-step convergence property; we'll discuss that.
- **Table D (ARR)** — best alpha is the one to report; if alpha=0 wins,
  ARR didn't help, report negative result.

**Save the printed cell output** (right-click → Save As, or just copy it into
a `day2_tables.md` file). You'll paste these tables into the paper.

---

## Step 5 — Update the manuscript with the new numbers

You should now have, locally on your laptop:

```
phase3_day2_eval_outputs/
  ├── headline_illum_s5/summary.csv          ← Table 1 row 1
  ├── headline_illum_s20/summary.csv         ← Table 1 row 2
  ├── method_ablation_baseline_s5/summary.csv ← Table 2 baseline row
  ├── step_ablation_full_ddim_s*/summary.csv ← Figure 3 + Table 3
  ├── arr_grid_arr_a*/summary.csv            ← Table 4 (ARR)
  ├── efficiency_t4_illum.csv                ← Table 1 efficiency cols
  ├── figure1_teaser_illum.pdf               ← drop into LaTeX
  └── figure3_step_ablation_illum.pdf        ← drop into LaTeX
```

The manuscript writing schedule:

- **Day 3 morning** — paste Table A as the new Table 1 (headline). Update
  the abstract's PSNR/SSIM/LPIPS/latency numbers. Use Figure 1 from this run.
- **Day 3 afternoon** — paste Table B as the method ablation. Use the
  paragraph templates in `NOVELTY_METHOD.md` for the writeup. Cite recent
  diffusion-LLIE baselines (concern 5 in `REVIEWER_RESPONSE.md`).
- **Day 4 morning** — Limitations + Reproducibility appendix
  + GitHub README + Broader Impact. (See `REVIEWER_RESPONSE.md` concerns
  8/9/10/11 — all are 30-min to 2-hour items.)
- **Day 4 afternoon** — final pass + submit.

---

## What if something breaks

### Cell 2 of training notebook says "Push your latest commits first"
You forgot Step 0. `git push` from your laptop, re-run Cell 2.

### Cell 3 of training notebook says "Could not find ... Real_captured/Train/Low"
The LOL-v2 dataset isn't attached, or you uploaded it without the
`Real_captured/` subfolder. Re-attach via right panel → + Add Data, or
re-upload making sure the directory tree is preserved.

### Cell 7 (training) crashes immediately with "out of memory"
Kaggle gave you a smaller GPU. Reduce `BATCH_SIZE` in Cell 5 from 4 to 2 and
re-run Cell 7.

### Validation PSNR is stuck below 18 dB after 30 epochs
Something is wrong with the training. Most likely cause: the dataset path
discovery in Cell 3 picked the wrong folder (e.g. it's training on eval15's
15 images instead of LOL-v2 Real's 689). Stop training, re-check Cell 3's
file counts. If they're wrong, set `DATASET_ROOT` manually in Cell 3.

### Eval notebook Cell 3 says "ILLUM_CKPT NOT FOUND"
You didn't upload the trained checkpoint as a Kaggle dataset (Step 3), or
you named it differently. Either re-upload as `lumidiff-day2-ckpts`, or
manually set `ILLUM_CKPT = "/kaggle/input/<your-dataset-name>/best_lolv2real_illum.pth"`
in Cell 3.

### LPIPS is empty in the printed tables
Cell 1's `pip install lpips` failed silently. Run `!pip install lpips` in a
fresh cell, then re-run Cells 5 and 11.

### The +illum model is *worse* than the Day 1 numbers
Possible. The Day 1 checkpoint may have been trained longer, on different
data, or with different augmentation. Three options:
1. Train for more epochs (set `EPOCHS = 200` in training Cell 5, retrain).
2. Report the +illum numbers as-is and frame the paper around the
   architecture+ablation contribution rather than headline-PSNR improvement.
3. Use the Day 1 checkpoint's numbers as the headline and the new training
   only for the method-ablation row.

Pick option 2 unless you have an extra full day for retraining.

---

## Submission-day final checklist (Day 4)

Before you click submit, verify each of these:

- [ ] Title doesn't contain "LuminaDiff" (unless rebranded with new framing).
- [ ] Abstract describes efficiency story, not absolute-quality story.
- [ ] Headline numbers in Table 1 are from the **new** +illum checkpoint, not Day 1.
- [ ] LPIPS column is filled in for every headline row.
- [ ] Latency is in **ms/image on T4**, not seconds on MPS.
- [ ] Figure 1 (teaser) is the *new* render from `kaggle_eval_illum`, not Day 1.
- [ ] Figure 3 (step ablation) shows both +illum and at least one ablation
      (baseline or vanilla DDIM) side by side.
- [ ] Method-ablation table (baseline vs +illum) is in Section 4.
- [ ] ARR ablation appears somewhere (even if best alpha = 0; report honestly).
- [ ] Recent diffusion-LLIE baselines cited with their published numbers
      (Diff-Retinex, PyDiff, LightenDiffusion, etc. — see concern 5).
- [ ] Reproducibility appendix lists all hyperparameters + the exact training
      command + checkpoint name + GitHub URL.
- [ ] Limitations section discusses LOL-v2 Synthetic weakness + at least one
      failure-case figure.
- [ ] Broader Impact paragraph mentions surveillance.
- [ ] Typo "It has been trained a U-net..." is fixed.
- [ ] GitHub repo has a `README.md` and is reachable from the paper's URL.

If all twelve boxes are ticked, you're submission-ready.
