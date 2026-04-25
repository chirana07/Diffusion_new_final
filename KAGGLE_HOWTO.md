# Running Day 1 on Kaggle — step by step

You've already done this once with `kaggle-diffusion.ipynb`. The new notebook (`kaggle_day1.ipynb`) follows the **same pattern**: pip install → `git clone` your repo → run scripts. The only differences are: it pulls more files, and it auto-detects your dataset paths.

Total time: ~30-60 min depending on T4 queue.

---

## Step 0 — Push your local changes to GitHub (one-time)

The Kaggle notebook clones from `https://github.com/chirana07/Diffusion_new_final.git` (same URL as your existing notebook). The new figure scripts are uncommitted on your laptop right now, so the clone won't see them yet.

On your laptop, in `New_dif/`:

```bash
git add evaluation.py ablate_steps.py \
        kaggle_day1.ipynb kaggle_day1.sh \
        make_comparison_grid.py make_step_ablation_figure.py make_teaser_figure.py \
        PHASE3_FEW_DAYS.md KAGGLE_HOWTO.md
git commit -m "Phase 3 day 1: figures + Kaggle runner"
git push
```

That's the only command you have to run locally.

---

## Step 1 — Open Kaggle and create a new notebook

1. Go to **https://www.kaggle.com/code**.
2. Click **+ New Notebook** (top right).
3. The new notebook opens in the editor.

---

## Step 2 — Set GPU

In the right-hand panel:

1. Click **Settings** (gear icon).
2. Find **Accelerator** → set it to **GPU T4 x2** (any T4 option works; one T4 is enough).
3. Find **Persistence** → set to **Variables and Files** (so outputs survive between sessions).
4. Click **Save**.

---

## Step 3 — Attach your three datasets

In the right-hand panel, click **+ Add Data** (or **+ Add Input**), then attach each of these. If they aren't already in your Kaggle account, upload them once (**Datasets → New Dataset → Upload**); after that they'll show up under "Your Datasets".

You need three things attached:

1. **LOL eval15** — folder with `eval15/low/` and `eval15/high/`
   - If you don't already have it on Kaggle: upload your local `New_dif/eval15/` folder.
2. **LOL-v2** — folder with `LOL-v2/Real_captured/Test/Low/` and `LOL-v2/Real_captured/Test/Normal/`
   - Upload your local `New_dif/LOL-v2/` folder if not already on Kaggle.
3. **Checkpoint** — your `final.pth` file
   - Upload `New_dif/checkpoints/last_pth_only/final.pth` as a Kaggle dataset.

After attaching, each dataset shows up as a folder under `/kaggle/input/<dataset-name>/`. **You don't need to look up the exact paths** — Cell 3 of the notebook auto-discovers them. If auto-discovery fails, the cell tells you exactly which one is missing and how to fix it.

---

## Step 4 — Paste the notebook content

Easiest way:

1. On your laptop, open `New_dif/kaggle_day1.ipynb` (e.g. in VSCode or JupyterLab).
2. In Kaggle's notebook editor, click **File → Import Notebook** (or drag-drop the `.ipynb` file onto the Kaggle page).

Alternative if import doesn't work:

1. Open `kaggle_day1.ipynb` in VSCode/Jupyter to read the cells.
2. In your new Kaggle notebook, create a code cell for each, copy-paste the cell contents in order.

Either way you should end up with 9 cells in your Kaggle notebook.

---

## Step 5 — Run

1. Click **Run All** (▶▶ button at the top).
2. Watch Cell 2 — if it prints `MISSING` next to any file, your `git push` from Step 0 didn't include those files. Fix locally, push, re-run Cell 2.
3. Watch Cell 3 — if any path comes back as `NOT FOUND`, the dataset isn't attached or it has a different folder layout. Either:
   - Click **+ Add Data** again and attach the missing dataset; or
   - In Cell 3, uncomment the relevant `MANUAL OVERRIDE` line and put in the actual path. (Right-click any file in the Kaggle right panel → **Copy path** to get the exact string.)
4. Watch Cell 4 — must say `cuda: True`. If not, you forgot to enable GPU (Step 2).
5. Cells 5-8 do the actual work. Total wall time on T4 is roughly:
   - Cell 5 (headline eval, 5 + 20 steps): ~5 min
   - Cell 6 (efficiency benchmark): ~2 min
   - Cell 7 (step ablation, 5 step counts): ~15 min
   - Cell 8 (figures): under 1 min
6. Cell 9 prints the headline table, prints the efficiency CSV, and bundles everything into `/kaggle/working/phase3_day1_outputs.zip`.

---

## Step 6 — Download

In the right-hand panel:

1. Click the **Output** tab.
2. Find `phase3_day1_outputs.zip`. Click the **download icon** next to it.
3. Unzip on your laptop. You'll have:
   - `headline_s5/summary.csv` — eval15 + lolv2_real PSNR/SSIM/**LPIPS** at 5 steps
   - `headline_s20/summary.csv` — same at 20 steps
   - `efficiency_t4.csv` — T4 latency / throughput / FLOPs
   - `figure1_teaser.pdf` + `.png`
   - `figure3_step_ablation.pdf` + `.png`
   - all the per-image CSVs and prediction PNGs

That's Day 1 done. You now have everything you need for Methods + Table 1 + Figures 1 and 3.

---

## Common problems

**Cell 2 prints `MISSING evaluation.py` (or any other file).**
You forgot to push, or the push didn't include the new files. On your laptop: `git status`, `git add`, `git commit`, `git push`. Then re-run Cell 2.

**Cell 2 fails with a Git LFS error on the checkpoint.**
That's why we attach the checkpoint as a separate Kaggle dataset (Step 3, item 3). The notebook reads it from `/kaggle/input/`, not from the cloned repo.

**Cell 3 says `EVAL15_LOW = NOT FOUND`.**
Your eval15 dataset on Kaggle may have a different folder structure than `eval15/low`. Click the dataset in the right panel, expand it, find the folder with the .png low-light images, right-click → **Copy path**. In Cell 3, uncomment the `EVAL15_LOW = ...` line and paste the path.

**Cell 4 says `cuda: False`.**
Accelerator wasn't set to GPU. Go to right panel → Settings → Accelerator → GPU T4 → Save. Click **Run All** again.

**The notebook session timed out during Cell 7.**
Kaggle free-tier sessions have a time limit. Run Cells 5, 6, 7 separately rather than `Run All`. Or: skip Cell 7 entirely (you already have the step-ablation data from your earlier MPS run; just upload that data as a Kaggle dataset and the figure script in Cell 8 will use it).

**LPIPS is still empty in the CSVs.**
Cell 1's `pip install lpips` failed silently. Run `!pip install lpips` in a fresh cell, then re-run Cells 5 and 9.
