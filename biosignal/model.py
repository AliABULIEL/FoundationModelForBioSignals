
import torch
import torch.nn as nn
import torch.nn.functional as F

class MBConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, expansion=4, stride=1):
        super().__init__()
        mid = in_ch * expansion
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, mid, 1, bias=False),
            nn.BatchNorm1d(mid), nn.SiLU(),
            nn.Conv1d(mid, mid, k, stride=stride, padding=k//2, groups=mid, bias=False),
            nn.BatchNorm1d(mid), nn.SiLU(),
            nn.Conv1d(mid, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch)
        )
        self.use_res = (in_ch == out_ch and stride == 1)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out

class EfficientNet1D(nn.Module):
    def __init__(self, in_ch=1, width_mult=1.0, emb_dim=256):
        super().__init__()
        base = int(32*width_mult)
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base), nn.SiLU()
        )
        cfg = [(2, 64, 1), (2, 128, 2), (3, 192, 2), (3, 256, 2)]
        layers = []
        ch = base
        for n, out_ch, s in cfg:
            out_ch = int(out_ch*width_mult)
            for i in range(n):
                layers.append(MBConv1D(ch, out_ch, stride=s if i==0 else 1))
                ch = out_ch
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(ch, emb_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=256, hid=1024, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.BatchNorm1d(hid), nn.ReLU(inplace=True),
            nn.Linear(hid, out_dim)
        )
    def forward(self, x): return self.net(x)
