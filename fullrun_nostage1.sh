#!/bin/bash

# Get the short git commit hash
COMMIT_HASH=$(git rev-parse --short HEAD 2>/dev/null)

# If git command fails (e.g. not in a git repo), use a fallback name
if [ -z "$COMMIT_HASH" ]; then
    echo "Warning: Could not get git commit hash. Using 'unknown'."
    COMMIT_HASH="unknown"
fi

# Set dataset type (default to massspecgym)
DATASET_TYPE=${1:-massspecgym}

# Set save directory
SAVE_DIR="checkpoints_align/${COMMIT_HASH}_${DATASET_TYPE}_nopretrain"

echo "========================================================"
echo "Starting Training Pipeline (without pre-trained model)"
echo "Target Directory: ${SAVE_DIR}"
echo "Dataset Type: ${DATASET_TYPE}"
echo "========================================================"

# 1. Train the alignment model
python train_align.py \
    --dataset_type "${DATASET_TYPE}" \
    --save_dir "${SAVE_DIR}"

# 2. Evaluate the aligned model
echo "========================================================"
echo "Starting Evaluation"
echo "========================================================"

python eval_align.py --dataset_type "${DATASET_TYPE}" --checkpoint "${SAVE_DIR}/best_model_stage2.pth"
