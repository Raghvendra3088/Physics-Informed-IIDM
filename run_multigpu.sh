#!/bin/bash
# Multi-GPU Orchestration Script for PI-LDM on A100 Server
# Environment: ml_env
# GPUs: 5x A100-PCIE-40GB

source /DATA/raghvendra3088/ml_env/bin/activate
cd /DATA/raghvendra3088/Physics-Informed-IIDM

mkdir -p logs

echo "Phase 1 Complete (Weights already saved). Launching Phase 2 & 3 (PI-LDM Seeds & Lambda Sweep) across GPUs 3 and 4..."

# GPU 3
(
    CUDA_VISIBLE_DEVICES=3 python src/training/train_pi_ldm.py --seed 42 --lambda_phys 0.05 > logs/pildm_seed42_L0.05.log 2>&1
    CUDA_VISIBLE_DEVICES=3 python src/training/train_pi_ldm.py --seed 42 --lambda_phys 0.5 > logs/pildm_seed42_L0.5.log 2>&1
    CUDA_VISIBLE_DEVICES=3 python src/training/train_pi_ldm.py --seed 456 --lambda_phys 0.05 > logs/pildm_seed456_L0.05.log 2>&1
    CUDA_VISIBLE_DEVICES=3 python src/training/train_pi_ldm.py --seed 123 --lambda_phys 0.05 > logs/pildm_seed123_L0.05.log 2>&1
) &

# GPU 4
(
    CUDA_VISIBLE_DEVICES=4 python src/training/train_pi_ldm.py --seed 42 --lambda_phys 1.0 > logs/pildm_seed42_L1.0.log 2>&1
    CUDA_VISIBLE_DEVICES=4 python src/training/train_pi_ldm.py --seed 42 --lambda_phys 0.001 > logs/pildm_seed42_L0.001.log 2>&1
    CUDA_VISIBLE_DEVICES=4 python src/training/train_pi_ldm.py --seed 42 --lambda_phys 0.01 > logs/pildm_seed42_L0.01.log 2>&1
    CUDA_VISIBLE_DEVICES=4 python src/training/train_pi_ldm.py --seed 42 --lambda_phys 0.1 > logs/pildm_seed42_L0.1.log 2>&1
) &

wait
echo "All Multi-GPU processes completed."
