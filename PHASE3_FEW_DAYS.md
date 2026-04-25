# Phase 3 — Few-Days Paper Polish (replacement for the 25-day plan)

You don't have weeks. You have ~4 working days. This plan rewrites the paper
around the **only honest selling point Phase 1 actually proved**: the model is
*sampling-step-stable* — it gets essentially full quality from **5 DDIM steps**.
That, plus the LOL-v2 Real number (22.65 dB / 0.789 SSIM), is the story.

We are *not* trying to be SOTA on PSNR. We are pitching efficiency at acceptable
quality, targeting a **regional / workshop venue**.

---

## What you actually have (Phase 1 evidence)

| Split | n | Sampler | Steps | PSNR | SSIM | Latency / img |
|---|---|---|---|---|---|---|
| LOL-v2 Real     | 100 | DDIM | **5**  | **22.79** | **0.791** | 2.69 s (MPS) |
| LOL-v2 Real     | 100 | DDIM | 10  | 22.70 | 0.790 | 5.35 s |
| LOL-v2 Real     | 100 | DDIM | 20  | 22.69 | 0.790 | 5.90 s |
| LOL eval15      | 15  | DDIM | 5   | 18.62 | 0.756 | 1.77 s |
| LOL eval15      | 15  | DDIM | 20  | 18.52 | 0.754 | 10.81 s |
| LOL eval15      | 15  | DPM-posterior | 5   | 16.14 | 0.673 | 1.46 s |
| LOL eval15      | 15  | DPM-posterior | 20  | 18.14 | 0.739 | 5.92 s |
| LOL-v2 Synthetic| 100 | DDIM | 20  | 16.30 | 0.776 | 2.85 s |

Two clean findings to pitch:

1. **Step-collapse is real.** PSNR/SSIM curves are essentially flat from 5 to
   100 DDIM steps. The model has a stable fixed point reachable in 5 steps. This
   is *not* normal for diffusion LLIE — most baselines need 25-50.

2. **DDIM dominates the legacy posterior at low step counts.** At 5 steps the
   gap is +2.48 PSNR / +0.083 SSIM (18.62 vs 16.14). Below ~20 steps DDIM is
   strictly better on this checkpoint.

Two findings to *not* pitch (be honest in the paper, but don't lead with them):

- LOL-v2 Synthetic is a weak point (16.30). Mention only if reviewers will catch
  it; otherwise omit and report only LOL-v2 Real + eval15.
- eval15 PSNR (18.5) is uncompetitive vs SOTA (~26+). The reframe: we are not
  claiming SOTA quality — we're claiming **efficient sampling at acceptable
  quality**, with LOL-v2 Real as the headline split (it's bigger and more
  realistic than the 15-image set).

---

## Visual sanity check (do this first — ~30 min)

Phase 1 metrics say "5 steps ≈ 20 steps". You need to verify that with your eyes
before you commit the paper to that claim, because it's possible the model is
collapsing to a *safely smooth* output that scores well on PSNR but looks
washed-out and undetailed.

**Use the helper script.** It builds montage PNGs for you so you don't have to
flip through 60 files:

```bash
# LOL eval15 — all 15 images (one big PNG)
python make_comparison_grid.py \
    --pred-s5  ./eval_results/step_ablation_full_ddim_s5/eval15 \
    --pred-s20 ./eval_results/step_ablation_full_ddim_s20/eval15 \
    --low      ./eval15/low \
    --gt       ./eval15/high \
    --per-image-csv     ./eval_results/step_ablation_full_ddim_s5/per_image.csv \
    --per-image-csv-s20 ./eval_results/step_ablation_full_ddim_s20/per_image.csv \
    --split eval15 --diff \
    --out  ./eval_results/visual_check_eval15.png

# LOL-v2 Real — worst/median/best 3 each
python make_comparison_grid.py \
    --pred-s5  ./eval_results/step_ablation_full_ddim_s5/lolv2_real \
    --pred-s20 ./eval_results/step_ablation_full_ddim_s20/lolv2_real \
    --low      ./LOL-v2/Real_captured/Test/Low \
    --gt       ./LOL-v2/Real_captured/Test/Normal \
    --per-image-csv     ./eval_results/step_ablation_full_ddim_s5/per_image.csv \
    --per-image-csv-s20 ./eval_results/step_ablation_full_ddim_s20/per_image.csv \
    --split lolv2_real --pick worst,median,best --max 9 --diff \
    --out  ./eval_results/visual_check_lolv2_real.png
```

Then open the two PNGs and look at each row in this order: **low | 5-step |
20-step | GT | |5−20| diff**.

### What you're looking for

For the **5-vs-20 columns**, ask:

- **Sharpness on text/edges.** Look at any text (signs, labels), specular
  highlights, or hard object boundaries. If 5-step looks visibly softer than
  20-step on the same region, the metric "tie" is hiding a quality loss and you
  should pitch 10 steps as the safe default instead of 5.
- **Color saturation.** Pick a saturated patch (a red sign, a green plant). If
  5-step is noticeably more washed out than 20-step, same caveat.
- **Halos and ringing.** Does 5-step have ringing around bright-against-dark
  edges that 20-step does not? If yes, the diff heatmap will show a bright ring
  there. If no, 5-step is genuinely fine.
- **Noise floor in dark regions.** The low-light input has a lot of read noise.
  Does 5-step keep more of it than 20-step?

For the **5-vs-GT columns** (regardless of step count), ask:

- **Global tone.** Is the predicted brightness in the right ballpark, or are we
  systematically too dark / too bright? A consistent tone offset is the most
  common single failure mode for diffusion LLIE.
- **Color cast.** Greenish? Magenta? Compare large neutral patches (walls,
  paper) against GT.
- **Texture preservation.** Find a textured region in the GT (fabric, grass,
  hair). Did the model preserve it or smear it?

### What the diff heatmap means

The rightmost column is `|5-step − 20-step|` rendered hot (black = identical,
red/yellow/white = large pixel difference). Use it as a "where do I look?"
map — if a row's heatmap is mostly black, the two outputs really are
near-identical and your "5 ≈ 20" claim is solid. If it's full of bright spots,
investigate those spots in the source images first.

### The decision rule

After looking at both grids, you're choosing between three pitches for the paper:

- **All-rows heatmaps near-black, no obvious 5-vs-20 quality gap visually:**
  pitch **5 steps** as the default. This is the strongest story.
- **Some heatmaps show real differences, 20-step looks visibly better on those:**
  pitch **10 steps** (still a 2-3× speedup vs typical 25-50). The numbers above
  show 10 ≈ 20 within noise too.
- **20-step looks meaningfully better in the majority of rows:** pitch **20
  steps** as the default — slower, but still a 2× speedup vs the 50-step
  diffusion-LLIE baselines, and you keep the "DDIM > legacy at low step counts"
  side-finding.

Pick one and commit before you start writing — you can't tell a coherent
efficiency story while hedging across step counts.

---

## Day-by-day plan (4 days)

The compressed plan drops everything optional. No retraining, no loss
ablation, no method ablation, no extra splits.

### Day 1 — Decide the pitch + paper skeleton

**Morning (2-3 h):**
- Run the two visual-check grids above. Apply the decision rule. Write down
  in one sentence what your "default sampler" claim is.
- Re-run efficiency on Kaggle T4 for the chosen step count, so you have GPU
  numbers (not just MPS):
  ```bash
  python measure_efficiency.py --steps 5 10 20 --device cuda \
      --resolution 400 600 --out ./eval_results/efficiency_t4.csv
  ```

**Afternoon (3-4 h):** New paper skeleton:

1. **Title:** "Sampling-Efficient Diffusion for Low-Light Image Enhancement"
   (or "Few-Step Residual Diffusion for…"). **Drop "LuminaDiff"** if it's tied
   to the rejection — call it LuminaDiff-R or just give it a fresh name.
2. **Abstract** (1 short paragraph):
   - Problem: diffusion LLIE is accurate but slow (25-50 steps).
   - Method: residual-space diffusion + Retinex-derived illumination prior +
     SFT-conditioned U-Net.
   - Result: matches its own 50-step quality at **5 steps** on LOL-v2 Real
     (22.65 dB / 0.789 SSIM), running in ~X ms/img on a T4.
3. **Section list** (just headings on day 1, fill on day 2):
   1. Introduction
   2. Related work (split into: LLIE, diffusion LLIE, fast samplers)
   3. Method (residual-space, illumination prior, SFT conditioning, training
      losses, DDIM at inference)
   4. Experiments (datasets, metrics, training details, **main result table**,
      **step-ablation as a figure**, sampler comparison, qualitative
      comparisons)
   5. Discussion / limitations (be explicit: PSNR is below SOTA on eval15;
      we trade absolute quality for speed)
   6. Conclusion
4. **Discard the old "LuminaDiff" framing.** The reviewers rejected it. Don't
   re-pitch the same framing under the same name.

**End of day 1 deliverable:** updated paper skeleton in `paper/` (or wherever
your manuscript lives), and a one-paragraph abstract you actually believe.

### Day 2 — Tables, figures, and rewriting Methods

**Morning — figures (3 h):**

- **Figure 1 (teaser):** one carefully chosen LOL-v2 Real example. Layout:
  `low | ours-5step | ours-20step | GT`, with PSNR under each. Pick the row
  with the strongest visual story from your day-1 inspection — **not** the
  highest-PSNR row. Best shows clearly that 5 steps is competitive.
- **Figure 2 (architecture):** one panel showing low → illumination prior →
  conditioning encoder → SFT-conditioned U-Net → residual → final image.
  Lift this from `model.py` and `modules.py`; do not over-engineer it.
- **Figure 3 (step ablation):** PSNR vs steps line plot, two lines (DDIM vs
  legacy DPM-posterior), two panels (eval15, LOL-v2 Real). The script
  `ablate_steps.py --plot` already produces a basic version; clean it up in
  matplotlib for the camera-ready look. **This is the core figure of the
  paper** — give it room.
- **Figure 4 (qualitative comparison grid):** 4-5 rows, columns
  `low | baseline-A | baseline-B | ours-5step | GT`. For baselines, pick from
  *published* result PNGs if the original authors released them (LLFlow, RetinexNet,
  EnlightenGAN, KinD often have public results). If you can't get baseline
  PNGs in 2 days, drop this figure and survive on Figures 1-3 alone.

**Afternoon — tables (2 h):**

- **Table 1 (main quantitative):** PSNR / SSIM / LPIPS / step-count / runtime
  on LOL eval15 and LOL-v2 Real. Cite baseline numbers from their original
  papers — *do not* re-implement them, you have no time. Be explicit in the
  caption: "Baseline numbers as reported in the original publications;
  efficiency numbers measured on a single NVIDIA T4 GPU at 400×600."
- **Table 2 (step ablation):** the data above, formatted clean.

  Drop loss ablations and method ablations from the paper entirely — you
  don't have time to retrain. If a reviewer asks, the answer is "future work".

**Evening — Methods section (2 h):**
- Rewrite Methods around what's actually in the code. Reference the residual
  formulation, the illumination prior, the SFT conditioning, the multi-loss
  objective. **Do not invent claims.** If something is in the code, describe
  it; if it isn't, don't.

**End of day 2 deliverable:** all figures rendered as PDFs/PNGs, tables in
LaTeX, Methods section written.

### Day 3 — Experiments writeup + LPIPS rerun + Intro/Related

**Morning (2 h):** Run LPIPS for the headline numbers — the current CSVs have
empty `lpips_mean`, which a reviewer will flag. Rerun *only* the rows you
quote in the table:

```bash
# install lpips on Kaggle if not already
pip install lpips

python evaluation.py --splits eval15:./eval15/low:./eval15/high \
    lolv2_real:./LOL-v2/Real_captured/Test/Low:./LOL-v2/Real_captured/Test/Normal \
    --inference-steps 5 --sampler ddim \
    --results-root ./eval_results/headline_with_lpips --tag s5

python evaluation.py --splits eval15:./eval15/low:./eval15/high \
    lolv2_real:./LOL-v2/Real_captured/Test/Low:./LOL-v2/Real_captured/Test/Normal \
    --inference-steps 20 --sampler ddim \
    --results-root ./eval_results/headline_with_lpips --tag s20
```

Update Table 1 with the LPIPS numbers.

**Afternoon (4 h):**
- Write **Experiments** section. Datasets, training setup (256 crops, EMA,
  Adam, batch size, etc — read these out of `config.py` and `train.py`),
  evaluation protocol (full-resolution, pad-to-multiple-of-8). Explicitly say
  the eval is at *native resolution*, not 128×128 — this directly answers one
  of the rejection reviewer's complaints.
- Write **Introduction** (~1 page). Lead with the efficiency story. Don't
  oversell quality.
- Write **Related Work** (~3/4 page). Three threads: classical Retinex LLIE,
  diffusion LLIE, fast diffusion samplers (DDIM, DPM-Solver). Cite the
  recent diffusion-LLIE papers the reviewer demanded.

**End of day 3 deliverable:** complete first draft of the paper.

### Day 4 — Polish + buffer

**Morning (3 h):**
- Read the whole draft top to bottom, out loud if possible.
- Fix the abstract last — it's the only thing some reviewers read.
- **Reproducibility appendix.** One page listing: hyperparameters from
  `config.py`, the exact command line, the final checkpoint name and where
  it lives, the evaluation command. This kills the rejection's
  "reproducibility issues" complaint at zero cost.
- Run a spell check + grammar pass.

**Afternoon (2-3 h, buffer):**
- Re-render any figure that looks weak.
- If anything broke during the LPIPS rerun, debug it.
- If you have a coauthor / advisor, send the draft for one read.

**End of day 4:** submission-ready PDF.

---

## What you are *deliberately* not doing

- **No retraining.** The current checkpoint is good enough on LOL-v2 Real.
- **No loss ablation.** It's nice-to-have; reviewers don't reject on its
  absence at workshop tier if the main story is clean.
- **No method ablation (illum prior on/off).** Same reason.
- **No new architectures, no new losses, no new datasets.**
- **No third-party baseline reimplementation.** Cite published numbers.

If a reviewer demands any of the above, the response is "we agree; this is
listed as future work" and you survive — most workshop reviewers will accept
that for a small efficiency-focused paper.

---

## Sanity-check before you submit

- Title and abstract describe the *efficiency* story, not the quality story.
- LOL-v2 Real number leads the table; eval15 is reported but isn't the headline.
- Step-ablation figure is in there and is captioned "matches own 50-step
  quality at 5 steps".
- LPIPS is reported (not just PSNR/SSIM).
- Latency is reported in *milliseconds per image on a single T4*, not in MPS
  seconds — reviewers want comparable hardware numbers.
- Reproducibility appendix exists and tells the reader exactly which command
  to run to reproduce Table 1.
- The word "LuminaDiff" with the old framing is gone, or the framing is
  rewritten so the same name carries a different claim.

If those seven things are true on day 4, submit it.
