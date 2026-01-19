#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "=========================================================="
echo "   Fruit Freshness Assessment System - Full Pipeline      "
echo "=========================================================="

# 1. Install Required Packages
echo "[1/5] Install Required Packages..."
pip install -r requirements.txt

# 2. Data Preprocessing
echo "[2/5] Running Data Preprocessing & Augmentation..."
python data_process.py

# 3. Label Encoding
echo "[3/5] Finalizing Label Mappings..."
python label_encoder.py

# 4. CNN Training
echo "[4/5] Training CNN-HSV Hybrid Model (Layer-wise Unfreezing)..."
# This step takes the most time
python train_CNN.py

# 5. SVM Ensemble
echo "[5/5] Extracting Embeddings & Training SVM Ensemble..."
python classifier_SVM.py

echo "=========================================================="
echo "✅ Pipeline Completed Successfully!"
echo "Check the 'result/' folder for performance reports and plots."
echo "=========================================================="