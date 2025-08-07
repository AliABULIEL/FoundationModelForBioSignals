
import torch, torch.nn as nn, torch.nn.functional as F
from .model import EfficientNet1D, ProjectionHead

class ContrastiveKoLeo(nn.Module):
    def __init__(self, t=0.04, lambda_k=0.1):
        super().__init__(); self.t=t; self.lk=lambda_k
    def forward(self, q, k):
        q = F.normalize(q, dim=1); k = F.normalize(k, dim=1)
        logits = q @ k.T / self.t
        labels = torch.arange(q.size(0), device=q.device)
        loss_nce = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        with torch.no_grad():
            d = q.size(1)
            mindist = torch.cdist(q, q).add(torch.eye(q.size(0), device=q.device)*1e9).min(dim=1)[0]
            loss_k = -torch.log(mindist**d + 1e-9).mean()
        return loss_nce + self.lk*loss_k

class SSLModel(nn.Module):
    def __init__(self, in_ch=1, emb_dim=256, proj_dim=128, ema=0.99):
        super().__init__()
        self.enc = EfficientNet1D(in_ch, emb_dim=emb_dim)
        self.proj = ProjectionHead(in_dim=emb_dim, out_dim=proj_dim)
        self.enc_m = EfficientNet1D(in_ch, emb_dim=emb_dim)
        self.proj_m = ProjectionHead(in_dim=emb_dim, out_dim=proj_dim)
        self.ema = ema
        self._init_m()
    def _init_m(self):
        for p_m, p in zip(self.enc_m.parameters(), self.enc.parameters()):
            p_m.data.copy_(p.data); p_m.requires_grad=False
        for p_m, p in zip(self.proj_m.parameters(), self.proj.parameters()):
            p_m.data.copy_(p.data); p_m.requires_grad=False
    @torch.no_grad()
    def _update_m(self):
        for p_m, p in zip(self.enc_m.parameters(), self.enc.parameters()):
            p_m.data.mul_(self.ema).add_(p.data*(1-self.ema))
        for p_m, p in zip(self.proj_m.parameters(), self.proj.parameters()):
            p_m.data.mul_(self.ema).add_(p.data*(1-self.ema))
    def forward(self, x1, x2):
        z1 = self.proj(self.enc(x1))
        with torch.no_grad():
            z2 = self.proj_m(self.enc_m(x2))
        return z1, z2
