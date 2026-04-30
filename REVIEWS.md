# All three reviews — verbatim

The submission `LuminaDiff: Efficient Diffusion-Based Low-Light Image Enhancement
with Multi-Loss Training` was rejected from ADScAI Conference 2026. Below are
both reviewer reviews and the area-chair meta-review, copied without edits.

---

## 1. Reviewer H2EM — Official Review of Submission 38

**Date:** 22 Apr 2026, 00:26 (modified 24 Apr 2026, 01:27)

**Paper Summary:** This paper presents LuminaDiff, a diffusion-based model for
low-light image enhancement that aims to balance perceptual quality and
computational efficiency. The approach leverages DDIM-style accelerated sampling
to reduce inference cost while maintaining reasonable enhancement quality. The
model is built on a U-Net denoising architecture and trained using a composite
multi-loss objective combining Charbonnier loss, SSIM loss, perceptual loss, and
total variation regularization. The evaluation is conducted on the LOL dataset,
specifically the eval15 subset, and results are reported using PSNR and SSIM
metrics.

**Summary Of Strengths:** The paper addresses a practically relevant problem,
namely making diffusion-based image enhancement feasible under real-world
constraints such as limited computational resources and latency requirements.
The motivation for using DDIM-style sampling to enable a controllable
quality–efficiency trade-off is sound and well aligned with current trends in
efficient generative modeling. The architecture and methodology are clearly
presented, and the use of a multi-objective loss function is appropriate for
the low-light image enhancement task. Additionally, the paper demonstrates
awareness of deployment considerations, which is a valuable aspect often
overlooked in similar work.

**Summary Of Weaknesses:** The evaluation is very limited, using only 15
images, which is insufficient for reliable conclusions. The results are not
competitive with stronger baselines, and this is not well discussed. The
novelty is limited, as the method combines existing techniques without a clear
new contribution. Key experimental details are missing, affecting
reproducibility. The efficiency claim is not supported by runtime or complexity
analysis. There are no ablation studies or comparisons with recent
diffusion-based methods.

**Comments / Suggestions / Typos:** The authors should evaluate on larger
datasets and include efficiency metrics such as runtime and model size.
Ablation studies on loss components and sampling steps are needed. Including
perceptual metrics and better qualitative comparisons would strengthen the
work. Minor language and clarity improvements are recommended.

**Limitations And Societal Impact:** Limitations are briefly mentioned but not
deeply analyzed. More discussion on failure cases and societal impact,
especially in surveillance contexts, is needed.

**Scores:**
- Reviewer Confidence: 4
- Soundness: 2.5
- Excitement: 2.5
- Overall Assessment: 2.5
- Reproducibility: 2
- Datasets: 2
- Software: 1
- Author Identity: 1
- Ethical Concerns: No / Needs Ethics Review: false
- Best Paper Justification: N/A

---

## 2. Reviewer XCNR — Official Review of Submission 38

**Date:** 14 Apr 2026, 00:30 (modified 24 Apr 2026, 01:27)

**Paper Summary:** This paper introduces LuminaDiff, a diffusion-based deep
learning model for low-light image enhancement (LLIE). To balance restoration
quality with the latency constraints of practical camera pipelines, the authors
employ a lightweight conditional U-Net architecture and utilize DDIM-like
strided sampling limited to 20 steps. The model is trained using a composite
loss function comprising Charbonnier, SSIM, Perceptual, and Total Variation
(TV) losses. The method is evaluated quantitatively using PSNR and SSIM on the
15-image "eval15" subset of the LOL dataset.

**Summary Of Strengths:**
- *Practical Motivation:* The paper targets a highly relevant and practical
  problem: the high inference latency of standard diffusion models, which
  currently bottlenecks their deployment in real-world, edge-device camera
  pipelines.
- *Comprehensive Loss Formulation:* The multi-loss objective is well-rounded,
  thoughtfully combining robust reconstruction (Charbonnier) with structural
  (SSIM), perceptual (VGG), and spatial smoothness (TV) constraints.
- *Clear Methodology:* The architectural choices and training pipeline are
  described clearly and are easy to follow.

**Summary Of Weaknesses:**
- *Missing Efficiency Metrics:* The most critical weakness is the lack of
  empirical evidence supporting the paper's primary claim. The title and
  abstract emphasize "Efficient Diffusion," but the authors provide zero
  quantitative computational metrics. There are no comparisons of inference
  time (e.g., seconds per image), model size (parameter count), or
  computational complexity (GFLOPs) against the baselines. Stating the use of
  "20 steps" is insufficient to prove efficiency.
- *Limited Evaluation Scope:* Evaluating the model exclusively on the 15
  images of the LOL eval15 subset is inadequate to demonstrate the model's
  robustness and generalizability.

**Comments / Suggestions / Typos:** Typo: In the abstract, "It has been trained
a U-net style denoiser" should be rephrased for grammatical correctness (e.g.,
"It trains a U-Net style denoiser").

**Scores:**
- Reviewer Confidence: 4
- Soundness: 2
- Excitement: 2
- Overall Assessment: 3
- Reproducibility: 3
- Datasets: 3
- Software: 2
- Author Identity: 1
- Limitations And Societal Impact: 2
- Ethical Concerns: None / Needs Ethics Review: false

---

## 3. Area Chair — Meta-Review

**Verdict:** This paper presents LuminaDiff, a diffusion-based model for
low-light image enhancement that aims to balance restoration quality with
practical efficiency constraints through DDIM-style accelerated sampling and
a lightweight U-Net denoiser.

Both reviewers agree that the problem is relevant and practically motivated.
Enabling diffusion models for real-world imaging pipelines is an important
direction. Reviewers also note that the methodology is clearly described and
that the composite loss function combining reconstruction, structural,
perceptual, and smoothness objectives is appropriate for the task.

However, both reviews raise major concerns about empirical validation. The
central claim of efficiency is not supported by runtime, parameter count,
FLOPs, or memory measurements. Reporting only a reduced number of sampling
steps does not establish practical efficiency. Evaluation is also very
limited, relying solely on the 15-image LOL eval15 subset, which is
insufficient to demonstrate robustness or generalisation. Reviewer H2EM
further notes that results are not clearly competitive with stronger
baselines, and **novelty appears modest since the paper largely combines
known components without a distinct technical advance**. Missing ablations
and limited comparison with recent diffusion-based LLIE methods further
weaken the submission.

Overall, while the motivation is sound, the current evidence does not
substantiate the claimed contributions.

- **Recommendation:** Reject
- **Confidence:** 4 (the area chair is confident but not absolutely certain)
- **Best Paper Award:** No
