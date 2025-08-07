
import torch, torch.nn as nn, torch.nn.functional as F
import csv, numpy as np, pathlib
from .model import EfficientNet1D

class LinearProbe(nn.Module):
    def __init__(self, encoder_ckpt, num_classes=2, task='classification'):
        super().__init__()
        self.enc = EfficientNet1D()
        self.enc.load_state_dict(torch.load(encoder_ckpt, map_location='cpu'))
        for p in self.enc.parameters(): p.requires_grad=False
        out_dim = 1 if task=='regression' else num_classes
        self.head = nn.Linear(256, out_dim)
        self.task = task
    def forward(self, x):
        with torch.no_grad():
            z = self.enc(x)
        return self.head(z)

def load_csv(csv_path):
    xs, ys = [], []
    with open(csv_path, newline='') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            xs.append(pathlib.Path(r['filepath']))
            ys.append(float(r['label']))
    return xs, ys
