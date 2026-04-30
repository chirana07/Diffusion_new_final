# DAY 2 v3 — Few-step Self-Distillation (the master runbook)

**This is the v3 plan.** From-scratch training overfit. Fine-tuning slowly
drifted away from the working solution. The v3 plan is **self-distillation**:
your Day 1 checkpoint becomes a frozen teacher running 5 DDIM substeps; a
trainable student (initialized from the same weights) learns to match the
teacher's multi-step refinement in a single forward pass. The student then
plugs back into the standard 5-step DDIM sampler at zero extra inference cost.

This is the cleanest, most defensible novelty story available within your
remaining time budget — citable foundations (Salimans & Ho 2022, Song et al.
2023), real training-time technique, addresses the area chair's "novelty
modest" verdict head-on.

Expected wall clock:
- ~20 min of your active time (push, set up notebook, kick off training)
- ~50–70 min unattended Kaggle GPU for training
- ~50–75 min for re-evaluation
- **Total: 1 active hour + 2 hours unattended = one afternoon**

---

## What changed since v2 (the fine-tune plan)

| | v2 fine-tune (failed) | v3 self-distillation |
|---|---|---|
| Init | Day 1 ckpt | Day 1 ckpt (both teacher and student) |
| Target | ground truth | **teacher's 5-substep DDIM refinement** + GT anchor |
| LR | 1e-5 | 5e-5 |
| Epochs | 50 | 20 |
| Per-iter cost | 1 fwd + 1 bwd | 6 fwd + 1 bwd (5 teacher substeps + student) |
| Wall time / run | 1.5 h | ~1 h |
| Story | architectural ablation | **training-time technique with citable foundation** |
| Risk if it doesn't beat teacher | "no improvement" | **still publishable** as honest analysis (FSD failed → teacher already near-optimal at this step count) |

---

## Step 0 — Push your local commits (do this FIRST)

The Kaggle notebook needs:
- `train_distill.py` — the new distillation training script
- updated `train.py` (already pushed — has `--init-from`)
- `kaggle_path_discovery.py` (already pushed)
- `kaggle_distill.ipynb` and `kaggle_eval_distill.ipynb`
- `NOVELTY_METHOD.md` (rewritten with FSD methods text)

```bash
cd "/Users/chirana/Desktop/UOM/Sem 3/Software project IFS/Diffusion_new/LUMIDIFF/New_dif"
git add .
git commit -m "Day 2 v3: few-step self-distillation training + eval notebooks"
git push
```

Verify on `https://github.com/chirana07/Diffusion_new_final` that the latest
commit is the one you just pushed.

---

## Step 1 — Confirm three datasets are on Kaggle

You'll attach all three to the distillation notebook in Step 2:

1. **LOL-v2** — must include `Real_captured/Train/` and `Real_captured/Test/`.
2. **LOL eval15** — for re-evaluation.
3. **Day 1 checkpoint** — your existing `final.pth`. This is what we distill *from*.

If any of these are missing, upload them as Kaggle datasets via **Datasets →
+ New Dataset**.

---

## Step 2 — Run the distillation notebook

### 2.1 New Kaggle notebook setup
1. **kaggle.com/code → + New Notebook**
2. Right panel → **Settings → Accelerator → GPU T4 x1 → Save**
3. Right panel → **Persistence → Variables and Files**
4. Right panel → **+ Add Data** three times → attach LOL-v2 + LOL eval15 + Day 1 checkpoint

### 2.2 Import + run
- Drag `kaggle_distill.ipynb` from your laptop onto the page.
- Click **Run All**.

| # | What | Wall time | Check |
|---|---|---|---|
| 1 | install | 30 s | `Successfully installed ...` |
| 2 | clone repo + verify files | 10 s | `Repo ready.` |
| 3 | preflight: discover datasets | 5 s | prints all candidates + validated counts (Train≥300, Test≥15) |
| 4 | locate Day 1 checkpoint | 1 s | prints `INIT_FROM = ...` and the size |
| 5 | GPU sanity | 1 s | `cuda: True ... Tesla T4` |
| 6 | **SMOKE TEST** (1 epoch + 5 val batches) | ~3–4 min | runs to completion; final line `Smoke test passed.` |
| 7 | print distillation config | instant | shows `lr=5e-5, teacher_steps=5, epochs=20` |
| 8 | **RUN — full distillation (20 epochs)** | ~50–70 min | per-epoch progress; val PSNR every 5 epochs should climb |
| 9 | bundle checkpoints + zip | 30 s | writes `day2_distill.zip` |

### 2.3 What good progress looks like (Cell 8)

The per-epoch output:

```
Distill epoch 0: 100%|██████████| 172/172 [01:00<00:00, ...]
[epoch 4] 5-step val PSNR 22.812  SSIM 0.7951  -> new best
Distill epoch 5: 100%|██████████| 172/172 [01:00<00:00, ...]
[epoch 9] 5-step val PSNR 23.214  SSIM 0.8003  -> new best
...
```

What to expect:
- **Epoch 0–4 val PSNR**: should be **close to your Day 1 baseline of 22.77 dB**
  (because the student starts as an identical copy of the teacher).
- **Epoch 5–10 val PSNR**: should *climb* if distillation is working. Realistic
  range: **23.0–24.0 dB**.
- **Epoch 15–20 val PSNR**: should plateau or keep climbing slowly. Best case:
  **24.0–25.0 dB**.

If val PSNR stays stuck at 22.7–22.8 dB throughout (i.e. essentially identical
to teacher), distillation is not adding anything and we're in the "honest
negative result" path — still publishable, just with the alternative paragraph
in `NOVELTY_METHOD.md`.

If val PSNR drops below 22.0 dB, **stop the cell**. Lower the LR (`DISTILL_LR =
"1e-5"` in Cell 7), reset the cell, retry. The other knob to try is
`W_ANCHOR = "1.0"` to pull harder toward GT.

### 2.4 Download checkpoint
- Cell 9 outputs `/kaggle/working/day2_distill.zip`.
- Output tab → download → unzip locally.
- You should have `best_distill_K5.pth` and `last_distill_K5.pth`.

### 2.5 Upload as a Kaggle dataset
- **Datasets → + New Dataset → upload `best_distill_K5.pth`** → name it
  `lumidiff-distill-ckpt` → set Private → Create.

---

## Step 3 — Run the re-evaluation notebook

### 3.1 New Kaggle notebook setup
Same as Step 2.1 but attach **four** datasets:
- LOL eval15
- LOL-v2
- Day 1 checkpoint (the teacher we'll compare against)
- `lumidiff-distill-ckpt` (the new student)

### 3.2 Import + run
- Drag `kaggle_eval_distill.ipynb` onto the page.
- Run All.

| # | What | Wall time |
|---|---|---|
| 1 | install | 30 s |
| 2 | clone repo | 10 s |
| 3 | preflight datasets | 5 s |
| 4 | locate student + teacher checkpoints | 1 s |
| 5 | GPU sanity | 1 s |
| 6 | headline eval (student @ 5/10/20 with LPIPS) | ~15 min |
| 7 | baseline (teacher @ 5 with LPIPS) | ~5 min |
| 8 | step ablation (5/10/20/50/100) | ~30 min |
| 9 | ARR grid (alpha 0–0.5) | ~5 min |
| 10 | T4 efficiency | ~2 min |
| 11 | render Figures 1 + 3 | 30 s |
| 12 | **print Tables A/B/C/D** | instant — copy into paper |
| 13 | bundle + zip | 30 s |

### 3.3 Read Cell 12 carefully

**Table A (headline, distilled student):** PSNR at 5 steps should be ≥ 22.77
(the teacher's number). Realistic positive case: **23.0–25.0 dB**.

**Table B (method ablation, teacher vs student):** the *method ablation row
that addresses the area chair's complaint*. The student should beat the
teacher by 0.3–2.0 dB if FSD is working.

**Table C (step ablation, student):** confirms the 5-step convergence claim
survives. Should be ~flat from 5 to 100 steps.

**Table D (ARR on student):** secondary contribution. Best alpha may be 0
(no improvement) — that's fine.

### 3.4 Download
- Output tab → `phase3_eval_distill_outputs.zip` → unzip locally next to
  the manuscript.

---

## Two outcomes and what to do for each

### Outcome A — Distillation works (student ≥ teacher + 0.5 dB)

**This is your strongest paper.** Pitch:

- Headline: distilled student at 5 DDIM steps achieves <X> dB PSNR / <Y> SSIM
  on LOL-v2 Real (vs the 22.77 teacher baseline) at 812 ms/image on T4.
- Novelty: FSD framework + ARR.
- Method ablation: FSD adds <X>+ dB.
- Step ablation: 5-step quality matches 50-step within 0.1 dB.

Use the *positive* ablation paragraph in `NOVELTY_METHOD.md`.

### Outcome B — Distillation doesn't help (student ≈ teacher)

**Still publishable.** Pitch:

- Headline: teacher at 5 DDIM steps achieves 22.77 dB on LOL-v2 Real, 812 ms/T4.
- Novelty: ARR + we *evaluated* FSD and report an honest negative result.
- The negative result IS a contribution: "the residual-space DDIM model with
  Retinex illumination prior is already near-optimal at 5 steps; further
  step-count reduction or quality gains require different architectural
  approaches."
- Step ablation: same finding.

Use the *negative* ablation paragraph in `NOVELTY_METHOD.md`.

Either way the paper has: efficiency benchmark + step ablation + LPIPS + ARR
+ FSD experiment (positive or negative). That's a full set of experimental
contributions.

---

## What to do if something breaks

### Cell 2 says "missing kaggle_path_discovery.py" or similar
You forgot Step 0 (push). On laptop: `git push`, then re-run Cell 2.

### Cell 3 prints "LOL-v2 Real TRAIN split not found"
LOL-v2 Train folder isn't on Kaggle. Re-upload with both Train and Test, or
manually override `discoveries['lolv2_real_train']` in Cell 3.

### Cell 6 (smoke test) fails with shape mismatch
The Day 1 checkpoint architecture doesn't match what `train_distill.py`
expects. This shouldn't happen — both teacher and student are built with
`use_illum_prior=False` matching the source ckpt. If it does, send me the
error.

### Cell 6 fails with OOM
Reduce batch size: in the smoke command, change `"--batch-size", "4"` to `"2"`.

### Cell 8 epoch-1 val PSNR is < 22 dB
The smart-load failed silently. Inspect the training startup output — the
`[smart_load] loaded=N partial=M skipped=K` line should show high `loaded`,
zero or one `partial`, near-zero `skipped`. If `loaded=0`, the checkpoint
keys don't match.

### Loss climbing during Cell 8
Lower LR: edit Cell 7, set `DISTILL_LR = "2e-5"`, re-run.
Or: increase anchor: `W_ANCHOR = "1.0"`.

### Val PSNR not improving over teacher (Outcome B detected early)
Don't panic. Two options:
1. Continue and accept the negative result. Still publishable.
2. Try `--teacher-steps 10` (deeper teacher refinement → potentially harder
   target → potentially better student). Edit Cell 7's `TEACHER_STEPS = 10`,
   re-run Cells 7 and 8. ~1.5x training time.

### LPIPS empty in eval table
`pip install lpips` in Cell 1 failed silently. Run `!pip install lpips` in a
fresh cell, then re-run eval Cells 6, 7, 12.

---

## Submission-day checklist (Day 4)

Tick every box before submission:

- [ ] Title doesn't lead with "LuminaDiff" (or has a refreshed framing).
- [ ] Abstract pitches the FSD + efficiency story.
- [ ] Headline numbers in Table 1 are from the **distilled student** (or the teacher if Outcome B).
- [ ] LPIPS column filled in for every headline row.
- [ ] Latency reported in **ms/image on a single T4**.
- [ ] Figure 1 (teaser) is the new render from `kaggle_eval_distill`.
- [ ] Figure 3 (step ablation) shows the few-step convergence holds.
- [ ] Method-ablation table (teacher vs distilled student) is in Section 4.
- [ ] FSD methods writeup (Section 3.X) is paste-in from `NOVELTY_METHOD.md`.
- [ ] ARR ablation table (Table D) somewhere.
- [ ] Recent diffusion-LLIE baselines cited (Diff-Retinex, PyDiff, LightenDiffusion).
- [ ] Reproducibility appendix lists the exact distillation command (`--init-from`, `--teacher-steps 5`, `--lr 5e-5`, `--epochs 20`).
- [ ] Limitations section discusses LOL-v2 Synthetic + at least one failure case.
- [ ] Broader Impact paragraph mentions surveillance.
- [ ] Typo "It has been trained a U-net..." is fixed.
- [ ] GitHub repo has a README.md.
- [ ] All 15 boxes ticked.

---

## File reference

| File | Purpose |
|---|---|
| `DAY2_GO.md` (this) | Master runbook — follow top to bottom |
| `train_distill.py` | Self-distillation training script |
| `kaggle_distill.ipynb` | Distillation notebook |
| `kaggle_eval_distill.ipynb` | Re-evaluation notebook |
| `kaggle_path_discovery.py` | Robust dataset auto-discovery |
| `train.py` | Has `--init-from`, `--lr`, `--dataset-root` |
| `evaluation.py` | Has `--gate-alpha`, `--gate-floor` |
| `diffusion.py` | Has ARR in `ddim_sample` |
| `NOVELTY_METHOD.md` | Section 3.X writeup for FSD + ARR + new contribution bullets |
| `REVIEWS.md` | All 3 reviews verbatim |
| `REVIEWER_RESPONSE.md` | 12 concerns mapped to actions |
| `phase3_day1_outputs/` | Day 1 results (kept for reference) |
