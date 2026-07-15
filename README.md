# Physics-Informed-IIDM
Physics-Informed-IIDM/
├── src/
│   ├── models/       ← physics_loss.py, gedi_denoiser.py, swin_encoder.py (baad mein), improved_iidm.py
│   ├── data/          ← dataset loaders (denoised GEDI wale)
│   ├── training/       ← train_physics_iidm.py
│   └── utils/           ← metrics, visualization (base se copy)
├── configs/            ← yaml/json configs (lambda_phys, lr, etc.)
├── checkpoints/         ← APNE hi naye checkpoints, base ke nahi
├── results/
├── logs/
└── notebooks/
