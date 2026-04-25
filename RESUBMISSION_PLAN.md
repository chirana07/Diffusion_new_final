# LuminaDiff v2 — 28-Day Resubmission Plan

**Author:** Chirana
**Current date:** 2026-04-24
**Target submission window:** 2026-05-15 to 2026-05-22 (3-4 weeks from today)
**Target venue tier:** Regional conference / workshop

---

## 0. Honest diagnosis of the rejection

Both reviewers agreed on a consistent picture. Ranked by severity:

1. **Numbers are not competitive.** Your PSNR of 17.12 is below eight of nine baselines in your own Table 2 — including 2018's Retinex-Net (19.51) and Zero-DCE (18.69), both of which are lighter than your model. Reviewer H2EM flagged this directly ("results are not competitive with stronger baselines"). **This is the rejection driver.** Everything else is secondary.
2. **"Efficient" is claimed but never measured.** No FLOPs, no params, no latency, no memory. Reviewer XCNR called this "the most critical weakness."
3. **Evaluation is 15 images.** Both reviewers flagged LOL eval15 alone as insufficient.
4. **No novelty hook.** The four-term loss is a sensible combination but not new. Reviewer H2EM: "the method combines existing techniques without a clear new contribution."
5. **No ablations.** Neither loss-component nor sampling-step ablations are reported.
6. **No recent diffusion-LLIE baselines.** LightenDiffusion, Diff-Retinex, PyDiff, GSAD are all missing.
7. **Reproducibility score 2-3.** Hyperparameters, training details, seeds not fully specified.
8. **Writing issues.** "It has been trained a U-net style denoiser" is the flagged one, but there are more. The narrative also over-claims.

**Implication:** A cosmetic resubmission (add ablations, add FLOPs, expand eval) will fail again because problem #1 stays unchanged. The plan below fixes #1 first, then #2-#8 in parallel.

---

## 1. The contribution pivot — what changes in the model

You asked whether a new architecture is worth it given the 4-day budget. It is, if we scope it correctly. Here is the minimal change that buys both a quality boost *and* a defensible novelty story:

### 1.1 Proposed method: **LuminaDiff-R — Retinex-conditioned residual diffusion**

Two specific changes to the existing pipeline:

**Change A — Residual / warm-start sampling (the big PSNR win).**
Today your sampler starts from pure Gaussian noise `x_T ~ N(0, I)` and denoises toward the clean image over 20 steps. That's a lot of work for 20 steps and explains much of the weak PSNR. Instead:

- Start sampling from the **low-light image itself** plus a small noise offset: `x_T = α·y_low + σ·ε`
- Train the denoiser to predict the **residual** (enhancement delta) rather than the full clean image: `ŷ = y_low + f_θ(y_low, t, prior)`

This pattern is used by recent fast-diffusion LLIE work (GSAD, PyDiff variants), and empirically pushes PSNR up by 2–4 dB on LOL. It is the single biggest lever you have.

**Change B — Retinex illumination prior as extra conditioning.**
For each input, cheaply compute an illumination map `L(y_low) = max_c(y_low[c]) then Gaussian-blur`. Concatenate it as an extra input channel to the U-Net. Zero added parameters of consequence (just one extra input channel in the first conv). This gives the denoiser explicit brightness information and is the "novel" hook you can claim.

### 1.2 Why this is a defensible contribution (not just a trick)

Frame the paper as: *"Retinex-conditioned residual diffusion for efficient LLIE — matching multi-step diffusion quality at 20 DDIM steps by injecting a cheap illumination prior and warm-starting from the input."*

The story has three pillars reviewers can latch onto:

- **Problem identification:** standard diffusion LLIE wastes sampling capacity learning to produce well-calibrated brightness from pure noise.
- **Method:** residual formulation + Retinex conditioning addresses the pathology directly.
- **Claim:** quality-competitive at 20 steps, verified by FLOPs/latency and step-count ablation.

That is a coherent story even for a regional venue. It is not CVPR-novel, but it is workshop-novel.

### 1.3 Implementation budget

| Task | Estimate |
|---|---|
| Modify sampler to warm-start from low-light input | 0.5 day |
| Switch denoiser head to predict residual | 0.5 day |
| Add illumination-map channel to U-Net input | 0.5 day |
| Retrain on LOL-v2 Real (Kaggle T4, ~12–18 hrs) | 1 day wall-clock |
| Eval on LOL eval15 + LOL-v2 Real + LOL-v2 Syn | 0.5 day |
| **Total** | **≈ 3 days** |

Leave 1 day of slack. If after 3 days PSNR has not moved from 17.12 into at least the high teens, fall back to path 1.2 below.

### 1.4 Fallback if the architectural change stalls

If by day 4 the new model is not clearly better, drop changes B and keep only A (residual warm-start), which alone should deliver most of the PSNR gain. If even A does not help, pivot the paper's framing:

- Retitle to *"LuminaDiff: A Practical Baseline for Diffusion-Based LLIE at 20 Steps"*
- Reposition as a reproducibility + deployment-analysis paper
- Lean hard into efficiency measurements, ablations, and evaluation breadth
- Be explicit: "we do not claim SOTA quality; we characterize the quality–efficiency envelope"

This is a weaker paper but it is internally consistent and reviewers will not flag the mismatch between claims and numbers.

---

## 2. Target venue shortlist

At workshop/regional tier with a ~30-day window and no travel constraint, good fits include:

- **BMVC 2026 workshops** — BMVC itself is mid-tier; its workshops accept shorter papers and often have May-June deadlines.
- **ICIP 2026** — IEEE Int'l Conference on Image Processing, typically has late-cycle deadlines; check the current CFP.
- **WACV 2027 workshops** — several vision workshops run on WACV with rolling deadlines.
- **Local/regional IEEE or ACM conferences** — ADScAI (your original venue) is one; find 2–3 similar regional ADS/AI venues in South/Southeast Asia.
- **CVPR 2026 workshops (e.g., NTIRE)** — NTIRE has a low-light track most years. Even if the main challenge deadline has passed, workshop paper tracks sometimes stay open longer.

**Action:** on day 1, do a focused search for venues with LLIE/restoration fit and deadlines between 2026-05-15 and 2026-06-15. Pick two — a primary and a backup — before writing starts.

---

## 3. Day-by-day schedule (28 days)

Dates assume start 2026-04-24.

### Week 1 (Apr 24 – Apr 30): model + data + venue

| Day | Date | Work |
|---|---|---|
| 1 | Apr 24 | Venue shortlist and pick. Set up Kaggle environment; push current codebase to a private Kaggle notebook. Verify LOL-v2 Real and Syn loaders. Reproduce the PSNR 17.12 number from the checkpoint. |
| 2 | Apr 25 | Implement change A (residual head + warm-start sampler). Sanity-test on a handful of LOL-v2 pairs with a short training run. |
| 3 | Apr 26 | Implement change B (Retinex illumination map channel). Start full retraining on LOL-v2 Real (Kaggle T4, overnight). |
| 4 | Apr 27 | Evaluate midpoint checkpoint on eval15. Decision point: if PSNR ≥ 19 dB, continue; if < 18, drop change B and retrain with only change A. |
| 5 | Apr 28 | Kick off a parallel training on LOL-v2 Syn. Start writing FLOPs/latency profiling code (fvcore + torch.profiler). |
| 6 | Apr 29 | Finalize one trained checkpoint per dataset (LOL-v2 Real, LOL-v2 Syn). Run headline eval: PSNR, SSIM, LPIPS on both test splits + LOL eval15 legacy. |
| 7 | Apr 30 | **Milestone: numbers locked for headline table.** If still uncompetitive, execute fallback (section 1.4). |

### Week 2 (May 1 – May 7): full experimental matrix

| Day | Date | Work |
|---|---|---|
| 8 | May 1 | Ablation 1: loss components. Train 4 variants removing one loss term each (Charbonnier-only, -SSIM, -perceptual, -TV). Small budget each — 1/3 epochs of full run. |
| 9 | May 2 | Ablation 2: sampling steps. Evaluate the best checkpoint at {5, 10, 20, 50, 100} DDIM steps. Plot PSNR vs. latency. |
| 10 | May 3 | Ablation 3: changes A and B individually (baseline, +A, +B, +A+B). |
| 11 | May 4 | Efficiency measurements: params (torchinfo), FLOPs per forward pass (fvcore), latency per image at each step count on T4 and M4 Pro CPU, peak memory. |
| 12 | May 5 | No-reference eval on unpaired sets: if you can grab LIME/NPE/DICM/MEF/VV (small, public), run NIQE and BRISQUE. |
| 13 | May 6 | Qualitative figure production: generate side-by-side comparisons on 6–8 representative images (1 extreme low-light, 2 moderate, 2 with color shift, 2 with heavy noise). |
| 14 | May 7 | **Milestone: all experiments complete.** Results frozen. |

### Week 3 (May 8 – May 14): writing

| Day | Date | Work |
|---|---|---|
| 15 | May 8 | Outline all sections. Draft abstract and introduction with the new contribution framing. |
| 16 | May 9 | Related work: rewrite section 2. Add and correctly cite LightenDiffusion, Diff-Retinex, PyDiff, GSAD, and at least one 2024–2025 diffusion-LLIE paper. |
| 17 | May 10 | Method section: write changes A and B clearly, with equations. Include a figure showing the residual-diffusion sampling path. |
| 18 | May 11 | Experiments section: headline tables, ablation tables, efficiency table, step-count curve figure. |
| 19 | May 12 | Results discussion + limitations + societal-impact paragraph (surveillance risks, dataset bias). |
| 20 | May 13 | Abstract rewrite, conclusion, polishing. |
| 21 | May 14 | **Milestone: full draft v1.** Send to a second reader if you have one. |

### Week 4 (May 15 – May 21): polish, reproducibility, submission

| Day | Date | Work |
|---|---|---|
| 22 | May 15 | Reproducibility package: requirements.txt, training configs, seed logs, eval script, checkpoint link. |
| 23 | May 16 | Address grammar and clarity passes. Run the draft through a grammar checker and a careful human read. Specifically fix the "It has been trained a U-net style denoiser" sentence and similar issues. |
| 24 | May 17 | Rebuttal-preparation checklist: for each reviewer complaint in the last round, write the specific sentence in the new paper that addresses it. |
| 25 | May 18 | Final figure polish. Verify every figure has a proper caption, legend, and is referenced in text. |
| 26 | May 19 | Format check against the target venue's template. Check page limits, reference style, anonymization. |
| 27 | May 20 | Final read-through. Tighten the abstract one more time. |
| 28 | May 21 | **Submit.** Leave day 28 free for formatting surprises. |

---

## 4. Experimental matrix required for acceptance

### 4.1 Datasets (headline table)

At minimum, evaluate on:

- **LOL eval15** — for continuity with the prior submission and apples-to-apples with older baselines.
- **LOL-v2 Real (100 test images)** — what reviewers will expect. This is the headline.
- **LOL-v2 Synthetic (100 test images)** — to show the model handles both distributions.

Strongly recommended in addition:

- **LIME + NPE + DICM + MEF + VV** — unpaired, run NIQE/BRISQUE for generalization argument. These are small (tens of images each); no training needed.

### 4.2 Baselines to include

You already have: LDR, LIME, Retinex-Net, URetinex-Net, EnlightenGAN, Zero-DCE, SCI, MSATr.

**Must add at least two recent diffusion-LLIE methods:**
- **Diff-Retinex** (ICCV 2023)
- **PyDiff** (IJCAI 2023)
- Or **LightenDiffusion** (ECCV 2024, which you already cite [5])
- Or **GSAD** (NeurIPS 2023)

You do not have to retrain these — use the numbers reported in their own papers on LOL, and cite them. Put "*" on those rows and note "results from original paper."

### 4.3 Ablations required

Three separate ablations, each with its own small table:

1. **Method components:** baseline, +residual warm-start (A), +Retinex prior (B), +A+B.
2. **Loss components:** full loss, −Charbonnier, −SSIM, −perceptual, −TV.
3. **Sampling steps:** {5, 10, 20, 50, 100} — PSNR/SSIM and latency for each. This is the main "efficiency" figure.

### 4.4 Efficiency table (the thing XCNR demanded)

A single table with:

| Method | Params (M) | FLOPs (G) | Latency per image (ms, T4) | Latency per image (ms, CPU) | Peak mem (MB) |
|---|---|---|---|---|---|

Include your method plus at least URetinex-Net and one diffusion baseline. Numbers for baselines can come from their papers or quick local runs.

### 4.5 Qualitative figures

Aim for two figures:

- A 4×6 grid: columns = {input, Zero-DCE, Retinex-Net, URetinex-Net, your method, ground truth}; rows = 4 representative images.
- A failure-case figure: 2–3 images where your method struggles, with honest discussion in the limitations section. Reviewers reward this.

---

## 5. Paper structure overhaul

Map each reviewer complaint to a specific section or addition:

| Reviewer complaint | Where it gets fixed |
|---|---|
| Weak PSNR | Method section 3 (changes A and B), Results table |
| No FLOPs/latency | New Efficiency subsection in Experiments |
| 15-image eval | Dataset subsection now lists LOL-v2 Real, Syn, + unpaired |
| No novelty | Abstract, intro, and a dedicated "Contributions" list that names three specific things |
| No ablations | New Ablation Studies section (three sub-tables) |
| No recent diffusion baselines | Related work paragraph on diffusion LLIE + baseline rows |
| Reproducibility | Appendix with hyperparameters, seeds, hardware, a GitHub link |
| Typo "It has been trained" | Day 23 polish pass |
| Limitations/societal impact | Expanded discussion section |

Use a sentence like "*To address prior feedback, we now evaluate on LOL-v2 (Real + Synthetic) and report runtime, parameters, and FLOPs*" somewhere in the intro. This signals to reviewers you have heard the feedback without literally saying "we were rejected before."

---

## 6. Reproducibility checklist

At the bottom of the paper (or in an appendix), list:

- Optimizer, learning rate, batch size, epochs, scheduler
- Loss weights λ_c, λ_s, λ_p, λ_tv
- Diffusion parameters: β schedule, number of training steps, number of inference steps, α and σ for warm-start
- U-Net configuration: channels per stage, attention blocks, parameter count
- Hardware: Kaggle T4, 12GB, CUDA version, PyTorch version
- Random seeds used
- Data preprocessing: crop size, augmentations, normalization range
- Dataset splits: exact image IDs used for train/val/test

Push a clean code repository (can be an anonymized GitHub Gist for double-blind venues). Reviewers explicitly gave you a 1/1 on "Software" — this is free marks.

---

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Architectural change does not yield PSNR improvement | Medium | High | Day 4 decision point; fall back to fallback plan (section 1.4) |
| Kaggle GPU quota exhausted mid-training | Medium | Medium | Checkpoint every epoch; resume possible. Save weights to Kaggle Datasets for persistence |
| LOL-v2 loader bugs (common cause of wrong numbers) | Medium | High | Day 1 reproduces PSNR 17.12 from your existing checkpoint before touching anything |
| Writing runs over time | High | Medium | Draft v1 by day 21 leaves 7 days of slack |
| Venue CFP disappears or deadline moves | Low | Medium | Pick primary + backup venue on day 1 |
| Format does not match venue template | Low | Low | Day 26 buffer |

---

## 8. What to keep from the current paper

Not everything in the current submission is wrong. Keep:

- The four-term loss motivation (it is a reasonable design and both reviewers acknowledged it)
- The preprocessing/output-scaling subsection (section 5.2) — this is a real technical detail that helps reproducibility
- Most of section 2 (related work), with additions
- Most of the figures' structure, with updated visuals

Discard or heavily rewrite:

- Section 6.1 positioning ("LuminaDiff strikes a good compromise") — this paragraph explicitly concedes the model is worse and frames it as a feature. Reviewers saw through it. Replace with concrete efficiency claims backed by measurements.
- The title may need a small adjustment if the contribution becomes Retinex-conditioning — e.g., "*LuminaDiff-R: Retinex-Conditioned Residual Diffusion for Efficient Low-Light Image Enhancement*."

---

## 9. Immediate next actions (today)

1. Shortlist three candidate venues with deadlines between May 15 and June 15; pick one primary and one backup.
2. Clone your current repo to a Kaggle notebook; confirm you can reproduce PSNR 17.12 on LOL eval15 before you change a single line.
3. Download LOL-v2 Real and Synthetic to the Kaggle workspace.
4. Sketch changes A and B on paper before coding — make sure the math makes sense for residual prediction under DDIM sampling.
5. Decide: am I willing to commit to the architectural change, or do I want to start from the fallback plan? Tell me before day 2 starts so we don't waste the 4-day architecture budget.

---

## Summary in one paragraph

Your rejection was driven by uncompetitive numbers, not by missing ablations. Adding tables to the current paper will get it rejected again. The realistic 28-day path is to make two small, defensible changes to your diffusion pipeline — warm-starting from the low-light input and adding a Retinex illumination channel — retrain on LOL-v2 instead of eval15, measure the efficiency you have been claiming, add the ablations and baselines reviewers asked for, rewrite the framing so the contribution is legible, and submit to a workshop or regional venue by May 21. If the architectural change does not pay off in four days, fall back to a reproducibility-and-deployment-analysis framing that is internally consistent with the current numbers.
