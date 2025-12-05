import argparse, os, yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import BackboneEncoder, LatentEncoder, LatentUNet, LatentDecoder, ProjectionHead
from dataset import SketchPhotoDataset
from sklearn.metrics import average_precision_score

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/default.yaml')
    p.add_argument('--checkpoint', type=str, required=True)
    return p.parse_args()

def load_model(cfg, ckpt_path, device):
    backbone = BackboneEncoder(backbone=cfg['model']['backbone']).to(device)
    lenc = LatentEncoder(in_dim=backbone.out_dim, latent_dim=cfg['model']['latent_dim']).to(device)
    unet = LatentUNet(latent_dim=cfg['model']['latent_dim'], depth=cfg['model']['unet_depth']).to(device)
    ldec = LatentDecoder(latent_dim=cfg['model']['latent_dim'], out_dim=backbone.out_dim).to(device)
    proj = ProjectionHead(latent_dim=cfg['model']['latent_dim'], embed_dim=cfg['model']['embed_dim']).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    ms = ck['model_state']
    backbone.load_state_dict(ms['backbone'])
    lenc.load_state_dict(ms['lenc'])
    unet.load_state_dict(ms['unet'])
    ldec.load_state_dict(ms['ldec'])
    proj.load_state_dict(ms['proj'])
    return backbone, lenc, unet, ldec, proj

def compute_embeddings(dataloader, models, device):
    backbone, lenc, unet, ldec, proj = models
    backbone.eval(); lenc.eval(); unet.eval(); ldec.eval(); proj.eval()
    embs = []
    labels = []
    with torch.no_grad():
        for sk, ph, lbl in dataloader:
            ph = ph.to(device)
            f_ph = backbone(ph)
            z_ph = lenc(f_ph)
            # simple one-step denoise
            pred = unet(z_ph, torch.zeros(z_ph.size(0)).to(device))
            z_rec = z_ph - pred
            emb = proj(z_rec).cpu().numpy()
            embs.append(emb)
            labels.extend(lbl.numpy().tolist())
    embs = np.vstack(embs)
    return embs, np.array(labels)

def evaluate(cfg, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, lenc, unet, ldec, proj = load_model(cfg, ckpt_path, device)
    ds = SketchPhotoDataset(cfg['dataset']['root'], cfg['dataset']['test_pairs'])
    dl = DataLoader(ds, batch_size=cfg['dataset']['batch_size'], shuffle=False, num_workers=4)
    embs, labels = compute_embeddings(dl, (backbone,lenc,unet,ldec,proj), device)
    # split queries and gallery by label assumption (user must prepare pairs accordingly)
    # For demonstration, treat half as queries and half as gallery
    n = embs.shape[0]
    q = embs[:n//2]; g = embs[n//2:]
    ql = labels[:n//2]; gl = labels[n//2:]
    # compute distances
    dists = np.linalg.norm(q[:,None,:] - g[None,:,:], axis=2)
    # compute mAP and P@K
    APs = []
    P_at_1 = 0
    for i in range(dists.shape[0]):
        ranks = np.argsort(dists[i])
        matches = (gl[ranks] == ql[i]).astype(int)
        if matches.sum() == 0:
            APs.append(0.0)
            continue
        # average precision
        cum = np.cumsum(matches)
        precision_at_k = cum / (np.arange(len(matches)) + 1)
        AP = (precision_at_k * matches).sum() / matches.sum()
        APs.append(AP)
        if matches[0]==1:
            P_at_1 += 1
    mAP = np.mean(APs)
    P1 = P_at_1 / dists.shape[0]
    print(f"mAP: {mAP:.4f}, P@1: {P1:.4f}")

if __name__ == "__main__":
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    evaluate(cfg, args.checkpoint)