#!/bin/bash

# Complete training pipeline

# Step 1: Pre-train on VitalDB
echo "Step 1: Pre-training on VitalDB..."
python main.py pretrain --modality ppg --ssl-method simsiam

# Get the latest checkpoint
PRETRAINED=$(ls -t data/outputs/checkpoints/vitaldb_*/encoder.pt | head -1)
echo "Using pretrained model: $PRETRAINED"

# Step 2: Fine-tune on BUT PPG
echo "Step 2: Fine-tuning on BUT PPG..."
python main.py finetune --modality ppg --ssl-method simsiam \
    --pretrained-path "$PRETRAINED"

# Get the fine-tuned checkpoint
FINETUNED=$(ls -t data/outputs/checkpoints/finetune_*/best_model.pt | head -1)
echo "Using fine-tuned model: $FINETUNED"

# Step 3: Evaluate
echo "Step 3: Evaluating..."
python main.py evaluate --modality ppg --checkpoint "$FINETUNED"