import argparse, os, yaml, random, math
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import BackboneEncoder, LatentEncoder, LatentUNet, LatentDecoder, ProjectionHead
from dataset import SketchPhotoDataset
from diffusion import cosine_beta_schedule, q_sample
from losses import triplet_margin_loss, l2_loss

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/default.yaml')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

def train(cfg):
    device = torch.device(cfg.get('device','cuda') if torch.cuda.is_available() else 'cpu')
    # models
    backbone = BackboneEncoder(backbone=cfg['model']['backbone']).to(device)
    lenc = LatentEncoder(in_dim=backbone.out_dim, latent_dim=cfg['model']['latent_dim']).to(device)
    unet = LatentUNet(latent_dim=cfg['model']['latent_dim'], depth=cfg['model']['unet_depth']).to(device)
    ldec = LatentDecoder(latent_dim=cfg['model']['latent_dim'], out_dim=backbone.out_dim).to(device)
    proj = ProjectionHead(latent_dim=cfg['model']['latent_dim'], embed_dim=cfg['model']['embed_dim']).to(device)

    params = list(backbone.parameters()) + list(lenc.parameters()) + list(unet.parameters()) + list(ldec.parameters()) + list(proj.parameters())
    opt = optim.AdamW(params, lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])

    # dataset
    ds = SketchPhotoDataset(cfg['dataset']['root'], cfg['dataset']['train_pairs'])
    dl = DataLoader(ds, batch_size=cfg['dataset']['batch_size'], shuffle=True, num_workers=4)

    T = cfg['model']['diffusion_steps']
    betas = cosine_beta_schedule(T)

    best_loss = 1e9
    for epoch in range(cfg['training']['epochs']):
        backbone.train(); lenc.train(); unet.train(); ldec.train(); proj.train()
        total_loss = 0.0
        for sk, ph, lbl in tqdm(dl, desc=f"Epoch {epoch}"):
            sk = sk.to(device); ph = ph.to(device)
            f_sk = backbone(sk); f_ph = backbone(ph)
            z_sk = lenc(f_sk); z_ph = lenc(f_ph)

            # sample a timestep for each sample
            batch_size = z_sk.size(0)
            t = torch.randint(0, T, (batch_size,)).to(device)
            # perform q_sample (approximate)
            z_sk_noisy, noise_sk = q_sample(z_sk, t, betas, device)
            z_ph_noisy, noise_ph = q_sample(z_ph, t, betas, device)

            # predict noise with unet (conditioning on t)
            t_norm = (t.float() / float(max(1, T-1))).to(device)
            pred_sk = unet(z_sk_noisy, t_norm)
            pred_ph = unet(z_ph_noisy, t_norm)

            diff_loss = l2_loss(pred_sk, noise_sk) + l2_loss(pred_ph, noise_ph)
            # reconstruct denoised latent
            z_sk_rec = z_sk_noisy - pred_sk
            z_ph_rec = z_ph_noisy - pred_ph

            recon_loss = l2_loss(ldec(z_sk_rec), f_sk) + l2_loss(ldec(z_ph_rec), f_ph)

            # embeddings and triplet
            emb_sk = proj(z_sk_rec)
            emb_ph = proj(z_ph_rec)
            # create negatives by shuffling
            neg_idx = torch.randperm(emb_ph.size(0)).to(device)
            emb_ph_neg = emb_ph[neg_idx]
            trip = triplet_margin_loss(emb_sk, emb_ph, emb_ph_neg, margin=cfg['training']['triplet_margin'])

            loss = cfg['loss_weights']['diff'] * diff_loss + cfg['loss_weights']['recon'] * recon_loss + trip

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dl)
        print(f"Epoch {epoch} avg_loss {avg_loss:.4f}")
        # checkpoint
        os.makedirs(cfg['logging']['save_dir'], exist_ok=True)
        ckpt = {
            'epoch': epoch,
            'model_state': {
                'backbone': backbone.state_dict(),
                'lenc': lenc.state_dict(),
                'unet': unet.state_dict(),
                'ldec': ldec.state_dict(),
                'proj': proj.state_dict()
            },
            'opt': opt.state_dict()
        }
        ckpt_path = os.path.join(cfg['logging']['save_dir'], f'ckpt_epoch_{epoch}.pth')
        torch.save(ckpt, ckpt_path)