# DiffReg-SBIR — Complete Reproducible Implementation (Option B)

This repository is a **full, production-ready** reference implementation for **DiffReg-SBIR**:
a latent-diffusion regularized framework for fine-grained sketch-based image retrieval.

**Key features**
- Full training loop with multi-step latent diffusion (configurable T).
- Cosine noise schedule and small lightweight Latent U-Net for denoising.
- Reconstruction consistency and joint metric learning (triplet).
- Evaluation scripts computing mAP, Precision@K, and CMC.
- Checkpointing, logging, reproducible config.
- Included figures from the paper (fig1_overview.jpg, fig2_arch.jpg).

**Quick start**
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Prepare datasets (Sketchy/TU-Berlin/QMUL). Place `train_pairs.txt` and `test_pairs.txt` in dataset root.
3. Edit `configs/default.yaml` to set dataset path, hyperparameters.
4. Train:
   ```
   python train.py --config configs/default.yaml
   ```
5. Evaluate:
   ```
   python eval.py --config configs/default.yaml --checkpoint ckpt_best.pth
   ```

Download the ZIP: [diffreg-sbir_repo_full.zip](sandbox:/mnt/data/diffreg-sbir_repo_full.zip)