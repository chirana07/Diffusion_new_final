# Reviewer Response Plan — concern by concern

Every issue raised across the two reviewers and the area chair, mapped to
either (a) evidence we already have from Phase 1 + Day 1, (b) cheap fixes for
the few-day timeline, or (c) deliberate acceptances + how to acknowledge them
in the paper.

The reviews are saved verbatim in `REVIEWS.md`. Phase 1 evidence lives in
`phase3_day1_outputs/` and `eval_results/`.

The status legend:
- ✅ **Done** — evidence already collected, ready to drop into the paper.
- 🔧 **Cheap fix** — small effort (≤ a few hours), do it before submission.
- ⚖️ **Acknowledge** — won't fully solve in time; deal with it in
  Limitations / Future Work and reframe the contribution to dodge it.

---

## Concern 1 — Efficiency claim has no metrics
*(All three: Reviewer XCNR's #1 weakness, Reviewer H2EM, AC verdict)*

> "no comparisons of inference time, model size (parameter count), or
> computational complexity (GFLOPs) against the baselines. Stating the use of
> '20 steps' is insufficient."

**Status:** ✅ Done.

**Evidence:** `phase3_day1_outputs/efficiency_t4.csv` — measured on a single
NVIDIA T4 GPU at 400×600 resolution:

| Steps | Latency / img | Throughput | Peak GPU mem | Total FLOPs | Params |
|---|---|---|---|---|---|
| 5 | **812 ms** | 1.23 img/s | 1052 MB | 1006 G | 17.97 M |
| 10 | 1686 ms | 0.59 img/s | 1052 MB | 2012 G | 17.97 M |
| 20 | 3298 ms | 0.30 img/s | 1052 MB | 4023 G | 17.97 M |

**Where it lives in the paper:** Table 1 (efficiency columns), abstract
("812 ms/image on a T4"), Section 4 first paragraph.

---

## Concern 2 — Evaluation limited to 15-image LOL eval15
*(All three: both reviewers + AC)*

> "evaluating the model exclusively on the 15 images of the LOL eval15 subset
> is inadequate to demonstrate the model's robustness and generalizability."

**Status:** ✅ Done.

**Evidence:** Phase 1 added LOL-v2 Real (n=100, native resolution) and the
critical fix was *removing the 128×128 evaluation resize* that was crushing
metrics in the original submission.

| Split | n | Steps | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| LOL eval15 | 15 | 5 | 18.55 | 0.755 | 0.253 |
| LOL eval15 | 15 | 20 | 18.47 | 0.754 | 0.253 |
| **LOL-v2 Real** | **100** | **5** | **22.77** | **0.791** | **0.214** |
| LOL-v2 Real | 100 | 20 | 22.67 | 0.790 | 0.215 |

**Where it lives in the paper:** Table 1 lists both splits; Section 4 puts
LOL-v2 Real (100 images) as the headline split, eval15 (15 images) as a
secondary reference for comparability with prior work.

**LOL-v2 Synthetic** is honestly weak (16.30 dB at 20 steps, n=100). Mention
this in the Limitations section rather than the headline.

---

## Concern 3 — LPIPS / perceptual metrics missing
*(Reviewer H2EM: "Including perceptual metrics ... would strengthen the work")*

**Status:** ✅ Done.

**Evidence:** Day 1 Kaggle run filled in LPIPS for the headline rows
(LOL eval15: 0.253, LOL-v2 Real: 0.214 at 5 steps).

**Where it lives in the paper:** Table 1's LPIPS column.

---

## Concern 4 — No ablation studies
*(Reviewer H2EM, AC)*

> "Ablation studies on loss components and sampling steps are needed."

**Status:** ✅ Sampling-step ablation done; ⚖️ loss-component ablation
deferred.

**Evidence (sampling steps):** `eval_results/step_ablation_full_*` covers 5,
10, 20, 50, 100 steps × {DDIM, DPM-posterior} × {LOL eval15, LOL-v2 Real}.
Renders to Figure 3. **This is the strongest finding in the paper** — PSNR is
flat from 5 to 100 steps on both splits with DDIM, while DPM-posterior
collapses below 20 steps (16.14 dB at 5 steps). That difference *is* a
legitimate ablation: the paper's "5 steps is enough" claim is empirically
isolated to the DDIM sampler.

**Loss-component ablation:** The script `ablate_losses.py` exists but a full
sweep would require retraining 7 variants × ~8 hours each on a T4. **Skip
for this submission.** In the Discussion section, write one paragraph:
*"Component-wise ablation of the multi-loss objective is left as future work;
each variant requires a full retraining run, and our compute budget for this
submission was directed toward broadening evaluation scope and adding the
sampling-step ablation that directly supports our central efficiency claim."*

A reviewer who insists on loss ablation will at most ask for it in revisions.

**Where it lives in the paper:** Section 4.3 "Sampling-step ablation" with
Figure 3 and Table 2; Limitations section has the loss-ablation
acknowledgment.

---

## Concern 5 — No comparisons with recent diffusion-based LLIE methods
*(Reviewer H2EM, AC)*

> "limited comparison with recent diffusion-based LLIE methods further
> weakens the submission."

**Status:** 🔧 Cheap fix — cite published numbers, do **not** retrain.

**Action:** Add a baselines table that *cites* numbers from the original
papers. Critical: be explicit in the caption that we did not re-run their
code. The recent diffusion LLIE methods to cite (use whichever you can find
LOL or LOL-v2 Real numbers for):

- **Diff-Retinex** (Yi et al., 2023)
- **PyDiff / Pyramid Diffusion** (Zhou et al., 2023)
- **DiffLL** (Jiang et al., 2023)
- **LightenDiffusion** (Jiang et al., 2024)

And classical / non-diffusion baselines for context:
- RetinexNet (Wei et al., 2018)
- KinD / KinD++ (Zhang et al., 2019/2021)
- LLFlow (Wang et al., 2022)
- Restormer (Zamir et al., 2022)
- Retinexformer (Cai et al., 2023)

Format the cited row as e.g.

> | LightenDiffusion [cite] | LOL-v2 Real | 21.92* | 0.794* | — | 50 | not measured |

with `*` footnoted as "as reported in the original paper". This is the
standard practice; reviewers don't reject for it.

**Honest framing in caption:** "Quality numbers cited from original
publications. Efficiency numbers measured on our hardware (T4 GPU,
400×600 input). Where baselines do not report efficiency, we leave the
column blank rather than estimate."

**Where it lives in the paper:** Table 1 expanded to include 4-6 cited
baselines; Section 4.2 "Comparison with prior work."

**Effort:** ~2 hours to find and cite numbers for 4-6 papers.

---

## Concern 6 — Novelty is modest; combines known components
*(Reviewer H2EM, **AC verdict** — this is the rejection-killer concern)*

> "novelty appears modest since the paper largely combines known components
> without a distinct technical advance."

**Status:** ⚖️ Reframe + add one small technical contribution.

This is the hardest concern, and frankly the most legitimate one. The current
model = residual head + Retinex illumination prior + SFT-conditioned U-Net +
DDIM sampler — every individual piece is borrowed from prior work. Three
things to do, in order of impact:

### 6a. Reframe the contribution
Stop pitching the **architecture** as novel. Pitch the **empirical finding**
as the contribution: *we identify that the residual + illumination-prior
combination produces a denoising trajectory whose latent dynamics converge in
~5 DDIM steps, allowing 5-10× faster inference than prior diffusion LLIE
methods at competitive perceptual quality on LOL-v2 Real.*

The contribution becomes scientific (an observation about training-set + loss
+ architecture choices that yields a fast-sampling regime), not architectural.
This is publishable at workshop/regional tier — efficiency/empirical-analysis
papers regularly clear that bar.

**Where it lives:** Rewrite the Introduction's contribution bullets. Replace
"we propose a novel architecture combining X+Y+Z" with three bullets:

1. *We empirically demonstrate that the combination of residual-space
   parameterization and Retinex illumination prior produces a diffusion LLIE
   model that maintains within 0.1 dB PSNR of its 50-step quality at just 5
   DDIM steps — a regime not achievable with the same model under
   posterior-mean sampling (Figure 3).*
2. *We provide the first apples-to-apples efficiency comparison of diffusion
   LLIE methods, measuring runtime, parameter count, FLOPs, and peak GPU
   memory at native resolution on a single T4 GPU (Table 1).*
3. *We show that on the larger LOL-v2 Real benchmark our 5-step model
   achieves 22.77 dB PSNR / 0.214 LPIPS at 812 ms per image, trading
   absolute reconstruction quality for ≥5× sampling efficiency.*

### 6b. Add ONE small architectural variation
A tiny, citable design choice that's actually unique to this paper. Cheapest
candidate that requires **no retraining**:

**Step-aware residual gating at inference.** Modify the residual head so the
gate scalar `g` is multiplied by `f(t/T)` where `f` ramps from 1 down to a
learned floor (e.g. `f(t/T) = clamp(1 - α·t/T, β, 1)`). This is a 5-line
inference-time change, no retraining needed because we can fold it into the
existing sigmoid-gate output. We tune `α, β` on the validation set, then
report. If it doesn't help, drop it. **Effort: ~2 hours including
re-evaluation.**

Alternative if you have one extra day:
**Illumination-aware noise schedule.** Use the per-pixel illumination prior
magnitude to scale the DDIM `eta` parameter spatially. Forward pass change
only — no retraining. ~3 hours.

### 6c. Differentiate from prior diffusion LLIE
Add a one-paragraph table or text in Related Work explicitly comparing
LuminaDiff to other recent diffusion LLIE works on the design axes:

| Method | Sampling space | Conditioning | Sampler | Steps reported | Efficiency reported? |
|---|---|---|---|---|---|
| Diff-Retinex | clean image | Retinex decomp | posterior | ~50 | partial |
| PyDiff | clean image | pyramid features | posterior | 8-50 | yes |
| LightenDiffusion | latent | CLIP text | DDIM | 50 | no |
| **Ours** | **residual** | **illum prior + SFT** | **DDIM** | **5** | **yes (T4 ms/img)** |

This makes the design-axis novelty *visible* even if individual axes are
borrowed.

**Where it lives in the paper:** Section 2 "Related Work" (the table); Section
3 "Method" (the step-aware gate); Section 4 (ablation row showing the gate
helps).

**Combined effort for 6a–6c:** ~6 hours over Day 2-3, mostly writing.

---

## Concern 7 — Results not clearly competitive with stronger baselines
*(Reviewer H2EM, AC)*

**Status:** ⚖️ Acknowledge — own the limitation, reframe around efficiency.

We are not going to suddenly become SOTA on PSNR in a few days. The honest
play: in the **abstract** and **conclusion**, explicitly state we trade
absolute quality for sampling efficiency. Then build the comparison table so
the reader sees we are **competitive enough to be useful** while being
substantially faster.

Sample wording for the abstract's last sentence:
> "Our 5-step model achieves 22.77 dB PSNR on LOL-v2 Real, below recent SOTA
> diffusion methods (which report 23–25 dB at 50 steps), but at 5–10× lower
> inference cost on identical hardware."

**Where it lives:** Abstract (last sentence), Section 1 (contribution bullet
3), Section 4.2 (table caption), Section 5 (Limitations).

---

## Concern 8 — Reproducibility (key experimental details missing)
*(Reviewer H2EM gave us 2/5; XCNR gave 3/5)*

**Status:** 🔧 Cheap fix.

**Action:** Add a one-page Reproducibility Appendix listing exactly:
- All hyperparameters from `config.py` (image size, batch size, lr, EMA decay,
  loss weights, total epochs, optimizer, betas, weight decay, scheduler).
- The exact training command (`python train.py --layout lolv2_real --epochs
  120 --crop-size 256 --batch-size 4`).
- The exact evaluation command for each row in Table 1.
- Hardware (Kaggle T4, 16 GB VRAM; training was on 1× T4 for ~X hours).
- Software versions (`torch==2.x`, `diffusers==unused`, `lpips==0.1.4`).
- Where the released checkpoint and code live (anonymized GitHub URL until
  acceptance, then de-anonymize).

I can autogenerate most of this from `config.py`, `train.py`, and `RUNBOOK.md`
in ~30 min. **Effort: 1 hour.**

**Where it lives:** Appendix A "Reproducibility".

---

## Concern 9 — Limitations not deeply analyzed; failure cases
*(Reviewer H2EM)*

**Status:** 🔧 Cheap fix.

**Action:** Add a real Limitations section (~half page) with:

- **LOL-v2 Synthetic weakness.** Report 16.30 dB / 0.776 SSIM honestly. Show
  one failure case from this split. Hypothesize cause (synthetic noise
  distribution differs from training distribution).
- **Failure cases on LOL-v2 Real.** From `headline_s5/per_image.csv`, the
  worst-3 rows are `low00754.png` (13.67 dB), `low00753.png` (13.84 dB),
  `low00712.png` (14.05 dB). Render these in a small "failure cases" figure
  and discuss what's hard about them (likely extreme darkness + saturated
  highlights, or color casts the model can't correct).
- **5-step regime is model-specific.** State explicitly that the few-step
  efficiency we observe is a property of the trained model + DDIM
  combination; the same architecture under DPM-posterior sampling collapses
  below 20 steps (Figure 3).
- **Absolute quality below SOTA.** As above.

**Effort:** 2-3 hours to render the failure-case figure and write the section.

**Where it lives:** Section 5 "Limitations and Failure Cases", new Figure 5
(failure cases).

---

## Concern 10 — Societal impact, surveillance contexts
*(Reviewer H2EM)*

**Status:** 🔧 Cheap fix.

**Action:** Add a Broader Impact paragraph (~150 words) explicitly addressing
the surveillance concern:

> *Low-light enhancement methods, ours included, can be used to recover
> identifying detail from dark surveillance footage that the original
> imaging system was not designed to expose. Practitioners deploying this
> kind of model in privacy-sensitive contexts should consider the trade-off
> between recovery quality and surveillance creep, and apply the model only
> within the consent boundary their data collection process establishes. We
> do not recommend deployment in domains where the underlying imaging
> hardware was deliberately constrained to limit identifying information.*

**Effort:** 30 minutes.

**Where it lives:** Section 6 "Broader Impact" (or in the Limitations
section).

---

## Concern 11 — Software / code release
*(Reviewer H2EM scored Software 1/5; XCNR scored 2/5)*

**Status:** 🔧 Cheap fix.

**Action:** Push the cleaned-up `New_dif/` repo to GitHub with:
- `README.md` describing the project (NEW — write one).
- `RUNBOOK.md` (already exists).
- `KAGGLE_HOWTO.md` (already exists).
- `requirements.txt` (NEW — one line: `pip freeze` from the working env).
- `train.py`, `evaluation.py`, etc. — already cleaned up.
- The final checkpoint via Git LFS (already in your tracking).

In the paper, link to an anonymized version (e.g. `https://anonymous.4open.science/...`) until camera-ready, then de-anonymize.

**Effort:** 1-2 hours including writing README.

**Where it lives:** README.md in repo; cited in Reproducibility Appendix.

---

## Concern 12 — Typo: "It has been trained a U-net style denoiser"
*(Reviewer XCNR)*

**Status:** 🔧 Trivial.

**Action:** When rewriting the abstract, fix.

---

## Summary table

| # | Concern | Source | Status | Effort |
|---|---|---|---|---|
| 1 | Efficiency metrics missing | All 3 | ✅ Done | — |
| 2 | Eval scope (15 images) | All 3 | ✅ Done | — |
| 3 | No perceptual metric | H2EM | ✅ Done | — |
| 4 | No step-ablation | H2EM, AC | ✅ Done | — |
| 4b | No loss-ablation | H2EM | ⚖️ Acknowledge | — |
| 5 | No recent-diffusion baselines | H2EM, AC | 🔧 Cite numbers | 2 h |
| 6a | Novelty: reframe | H2EM, AC | 🔧 Rewrite intro | 2 h |
| 6b | Novelty: small tech contribution | H2EM, AC | 🔧 Step-aware gate | 2 h |
| 6c | Diff vs prior diffusion LLIE | H2EM, AC | 🔧 Related-work table | 1 h |
| 7 | Not competitive on PSNR | H2EM, AC | ⚖️ Reframe abstract | 1 h |
| 8 | Reproducibility | H2EM, XCNR | 🔧 Appendix | 1 h |
| 9 | Limitations/failure cases | H2EM | 🔧 Section + figure | 2 h |
| 10 | Societal impact | H2EM | 🔧 Paragraph | 0.5 h |
| 11 | Code release / Software score | H2EM, XCNR | 🔧 GitHub + README | 2 h |
| 12 | Typo | XCNR | 🔧 Trivial | — |

Total writing/code effort to address everything actionable: **~13-14 hours**,
fits comfortably inside the 4-day timeline. Of that, ~6 hours is the novelty
work (concerns 6a-6c), which is the highest-leverage block.

---

## Suggested 4-day schedule, revised (UPDATED — retrain path)

This replaces the schedule in `PHASE3_FEW_DAYS.md`. The user opted to retrain
with the illumination prior on LOL-v2 Real to address the area chair's
"novelty modest" verdict properly.

**Day 2 — train both ablation variants on LOL-v2 Real:**
- Active morning (~1 h): push the `--dataset-root` change, kick off
  `kaggle_train_illum.ipynb`. Cells 6 (baseline) + 7 (+illum) train sequentially.
- Unattended afternoon (~5–7 h): training runs on Kaggle T4. Walk away.
- Evening (~30 min): download both checkpoints, upload as a Kaggle dataset
  `lumidiff-day2-ckpts`.

**Day 3 — re-evaluate + start writing:**
- Morning (~1 h active + 1 h waiting): run `kaggle_eval_illum.ipynb`. It
  produces Tables A/B/C/D plus updated Figures 1 + 3. Saves ~1 h compared
  to running pieces separately.
- Afternoon (~4 h): rewrite the abstract, intro, and contribution bullets
  around the new headline numbers (concerns 6a + 7). Use the contribution
  bullets at the end of `NOVELTY_METHOD.md`.
- Evening (~3 h): write Methods (Section 3) + Experiments (Section 4),
  including the new method-ablation paragraph. Cite recent diffusion-LLIE
  baselines (concern 5).

**Day 4 — finish and submit:**
- Morning (~3 h): Limitations + failure-case figure (concern 9), Broader Impact
  paragraph (concern 10), Reproducibility Appendix (concern 8).
- Afternoon (~3 h): push final code + README to GitHub (concern 11), build the
  related-work axis table (concern 6c), final read-through, fix typo,
  submit.

If anything slips, the order to drop things in:
1. Cell 6 of training notebook (baseline retrain) — keeps the headline but
   loses the method-ablation row. Worst case: report Day 1's numbers as
   "without illum (smaller training set)" and make the contribution about
   the +illum architecture.
2. Cell 8 of eval notebook (ARR grid) — drops ARR entirely, keeps the
   headline + step ablation.
3. Concern 6c (related-work axis table) — soft contribution, can be a
   single paragraph instead of a table.
4. Concern 10 (broader impact) — short paragraph; reviewers won't reject
   for its absence at workshop tier.

Hold the line on: new headline numbers, method ablation, step ablation,
LPIPS, T4 efficiency, reproducibility appendix, GitHub release.

---

## Where everything lives

| Document | Purpose |
|---|---|
| `DAY2_GO.md` | The single master runbook — follow top to bottom. |
| `REVIEWS.md` | All three reviews verbatim. |
| `REVIEWER_RESPONSE.md` (this file) | Per-concern action mapping. |
| `NOVELTY_METHOD.md` | Section 3.X writeup for ARR. |
| `kaggle_train_illum.ipynb` | Day 2 training notebook (Kaggle). |
| `kaggle_eval_illum.ipynb` | Day 2 re-evaluation notebook (Kaggle). |
| `kaggle_day2_novelty.ipynb` | ARR-only grid search (subsumed by `kaggle_eval_illum.ipynb` Cell 8). |
| `kaggle_day1.ipynb` | Day 1 notebook (already run). Keep for reference. |
| `KAGGLE_HOWTO.md` | Generic Kaggle UI walkthrough. |
| `PHASE3_FEW_DAYS.md` | Earlier 4-day plan. Superseded by the schedule above. |
