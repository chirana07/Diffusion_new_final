#!/usr/bin/env bash
# kaggle_day1.sh — equivalent of kaggle_day1.ipynb as a single shell script.
# Runs LPIPS-enabled headline eval, T4 efficiency benchmark, and renders Figures 1 and 3.
#
# Usage on Kaggle (in a Notebook cell or after `!bash kaggle_day1.sh`):
#   1. Edit the PATHS section below to match your Kaggle dataset mount points.
#   2. Make sure GPU is enabled (Kaggle right panel → Accelerator → GPU T4 x1).
#   3. Run.

set -euo pipefail

# ===================== PATHS — edit these =====================
# Where your code lives on Kaggle (path to a folder containing evaluation.py etc.)
SRC_DIR="/kaggle/input/lumidiff-src/New_dif"

# Datasets
EVAL15_LOW="/kaggle/input/lol-dataset/eval15/low"
EVAL15_HIGH="/kaggle/input/lol-dataset/eval15/high"
LOLV2_LOW="/kaggle/input/lol-v2/LOL-v2/Real_captured/Test/Low"
LOLV2_HIGH="/kaggle/input/lol-v2/LOL-v2/Real_captured/Test/Normal"

# Trained checkpoint
CHECKPOINT="/kaggle/input/lumidiff-checkpoint/final.pth"

# Working directory — outputs land in $WORK/eval_results/
WORK="/kaggle/working"
EVAL_RESULTS="$WORK/eval_results"
# ==============================================================

mkdir -p "$EVAL_RESULTS"
DST="$WORK/src"
rm -rf "$DST"
cp -r "$SRC_DIR" "$DST"
cd "$DST"

echo "==> Installing LPIPS + fvcore"
pip install lpips fvcore --quiet

echo "==> Verifying GPU"
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

SPLITS=(
    "eval15:${EVAL15_LOW}:${EVAL15_HIGH}"
    "lolv2_real:${LOLV2_LOW}:${LOLV2_HIGH}"
)

# -------- Job 1: headline eval at 5 and 20 steps with LPIPS --------
for STEPS in 5 20; do
    echo "==> Headline eval, DDIM ${STEPS} steps"
    python evaluation.py \
        --splits "${SPLITS[@]}" \
        --inference-steps "$STEPS" \
        --sampler ddim \
        --checkpoint "$CHECKPOINT" \
        --results-root "$EVAL_RESULTS/headline" \
        --tag "s${STEPS}"
done

# -------- Job 2: T4 efficiency --------
echo "==> T4 efficiency benchmark"
python measure_efficiency.py \
    --steps 5 10 20 \
    --resolution 400 600 \
    --device cuda \
    --checkpoint "$CHECKPOINT" \
    --out-csv "$EVAL_RESULTS/efficiency_t4.csv"

# -------- Job 3a: Figure 1 (teaser) --------
echo "==> Figure 1 (teaser)"
python make_teaser_figure.py \
    --pred-s5  "$EVAL_RESULTS/headline_s5/lolv2_real" \
    --pred-s20 "$EVAL_RESULTS/headline_s20/lolv2_real" \
    --low      "$LOLV2_LOW" \
    --gt       "$LOLV2_HIGH" \
    --per-image-csv     "$EVAL_RESULTS/headline_s5/per_image.csv" \
    --per-image-csv-s20 "$EVAL_RESULTS/headline_s20/per_image.csv" \
    --split lolv2_real \
    --out     "$EVAL_RESULTS/figure1_teaser.pdf" \
    --out-png "$EVAL_RESULTS/figure1_teaser.png"

# -------- Job 3b: step-ablation (recompute on T4 if not present, then plot) --------
RECOMPUTE_STEP_ABLATION="${RECOMPUTE_STEP_ABLATION:-1}"  # 1 = recompute, 0 = skip
if [[ "$RECOMPUTE_STEP_ABLATION" == "1" ]]; then
    for N in 5 10 20 50 100; do
        echo "==> Step ablation, DDIM ${N} steps"
        python evaluation.py \
            --splits "${SPLITS[@]}" \
            --inference-steps "$N" \
            --sampler ddim \
            --checkpoint "$CHECKPOINT" \
            --results-root "$EVAL_RESULTS/step_ablation_full" \
            --tag "ddim_s${N}" \
            --no-lpips
    done
fi

echo "==> Figure 3 (step ablation)"
python make_step_ablation_figure.py \
    --eval-root "$EVAL_RESULTS" \
    --out     "$EVAL_RESULTS/figure3_step_ablation.pdf" \
    --out-png "$EVAL_RESULTS/figure3_step_ablation.png"

# -------- Bundle outputs --------
echo "==> Zipping outputs"
cd "$WORK"
rm -f phase3_day1_outputs.zip
( cd "$WORK" && zip -r phase3_day1_outputs.zip eval_results > /dev/null )
echo "Wrote $WORK/phase3_day1_outputs.zip"

echo
echo "Done. Headline summary:"
for TAG in s5 s20; do
    P="$EVAL_RESULTS/headline_${TAG}/summary.csv"
    if [[ -f "$P" ]]; then
        echo "--- $P ---"
        cat "$P"
    fi
done
echo "--- $EVAL_RESULTS/efficiency_t4.csv ---"
cat "$EVAL_RESULTS/efficiency_t4.csv" || true
