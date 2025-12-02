#!/bin/bash
# Default training script for Masked Diffusion Model - Test Run
# This script runs a quick test to verify the training setup works correctly

echo "=========================================="
echo "Masked Diffusion Model - Test Training Run"
echo "=========================================="
echo ""

# Set default parameters for test run
BATCH_SIZE=32
EPOCHS=5
LEARNING_RATE=1e-4
IMAGE_SIZE=32
SAVE_EVERY=5
SAMPLE_EVERY=2

# Optional: Override with command line arguments
if [ ! -z "$1" ]; then
    BATCH_SIZE=$1
fi
if [ ! -z "$2" ]; then
    EPOCHS=$2
fi
if [ ! -z "$3" ]; then
    LEARNING_RATE=$3
fi

echo "Configuration:"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Image size: $IMAGE_SIZE"
echo "  Save checkpoint every: $SAVE_EVERY epochs"
echo "  Generate samples every: $SAMPLE_EVERY epochs"
echo ""
echo "Starting training..."
echo ""

# Run training
python main.py --mode train \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --image_size $IMAGE_SIZE \
    --save_every $SAVE_EVERY \
    --sample_every $SAMPLE_EVERY \
    --data_dir ./data \
    --save_dir ./checkpoints \
    --sample_dir ./samples \
    --device cpu

echo ""
echo "=========================================="
echo "Test training completed!"
echo "Check ./checkpoints/ for saved models"
echo "Check ./samples/ for generated samples"
echo "=========================================="

