# DiffReg-SBIR 



**Key features**
- Full training loop with multi-step latent diffusion (configurable T).
- Cosine noise schedule and small lightweight Latent U-Net for denoising.
- Reconstruction consistency and joint metric learning (triplet).
- Evaluation scripts computing mAP, Precision@K, and CMC.



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

📂 Datasets

- **ShoeV2 / ChairV2**  
  [Sketchy Official Website](https://sketchx.eecs.qmul.ac.uk/downloads/)  
  [Google Drive Download](https://drive.google.com/file/d/1frltfiEd9ymnODZFHYrbg741kfys1rq1/view)

- **Sketchy**  
  [Sketchy Official Website](https://sketchx.eecs.qmul.ac.uk/downloads/)  
  [Google Drive Download](https://drive.google.com/file/d/11GAr0jrtowTnR3otyQbNMSLPeHyvecdP/view)

- **TU-Berlin**  
  [TU-Berlin Official Website](https://www.tu-berlin.de/)  
  [Google Drive Download](https://drive.google.com/file/d/12VV40j5Nf4hNBfFy0AhYEtql1OjwXAUC/view)

   
Citation: If you use this code, please cite:

title = {Cross-Modal Spectral–Spatial Transformer for Fine-Grained SBIR},

author = {Mohammed A. S. Al-Mohamadi and Prabhakar C. J.},

journal = {.......}, year = {2025} }

License: This project is released under the MIT License.

Contact: almohmdy30@gmail.com GitHub: https://github.com/mohammedalmohmdy
