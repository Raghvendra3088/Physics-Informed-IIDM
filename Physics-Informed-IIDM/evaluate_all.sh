#!/bin/bash
source ../venv/bin/activate
for d in results/checkpoints/pildm_seed42_L*/; do
    echo "Evaluating $d"
    python src/training/evaluate_pi_ldm.py --ckpt_dir "$d"
done
