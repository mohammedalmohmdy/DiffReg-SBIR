import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BackboneEncoder(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()
        if backbone == 'resnet50':
            m = models.resnet50(pretrained=pretrained)
            self.features = nn.Sequential(*list(m.children())[:-2])  # up to conv5_x
            self.pool = nn.AdaptiveAvgPool2d((1,1))
            self.out_dim = 2048
        else:
            raise NotImplementedError

    def forward(self, x):
        f = self.features(x)
        v = self.pool(f).flatten(1)
        return v  # [B, out_dim]

class LatentEncoder(nn.Module):
    def __init__(self, in_dim=2048, latent_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim//2, latent_dim)
        )

    def forward(self, x):
        return self.fc(x)

class LatentDecoder(nn.Module):
    def __init__(self, latent_dim=256, out_dim=2048):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, out_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim//2, out_dim)
        )
    def forward(self, z):
        return self.fc(z)

# Latent U-Net: small MLP U-Net style
class LatentUNet(nn.Module):
    def __init__(self, latent_dim=256, hidden=512, depth=3, time_embed_dim=128):
        super().__init__()
        self.depth = depth
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.encs = nn.ModuleList()
        self.decs = nn.ModuleList()
        in_ch = latent_dim
        for i in range(depth):
            self.encs.append(nn.Sequential(
                nn.Linear(in_ch if i==0 else hidden, hidden),
                nn.ReLU(inplace=True)
            ))
        for i in range(depth):
            self.decs.append(nn.Sequential(
                nn.Linear(hidden*2 if i==0 else hidden, hidden),
                nn.ReLU(inplace=True)
            ))
        self.out = nn.Linear(hidden, latent_dim)

    def forward(self, z, t):
        # t: scalar or tensor [B,1] (normalized 0..1)
        te = self.time_mlp(t.view(-1,1))
        h = z
        skips = []
        for enc in self.encs:
            h = enc(h + te) if te is not None else enc(h)
            skips.append(h)
        for i, dec in enumerate(self.decs):
            if i==0:
                h = torch.cat([h, skips[-1]], dim=1)
                h = dec(h)
            else:
                h = dec(h)
        return self.out(h)

class ProjectionHead(nn.Module):
    def __init__(self, latent_dim=256, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
    def forward(self, z):
        x = self.net(z)
        return F.normalize(x, dim=1)