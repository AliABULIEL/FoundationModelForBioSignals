
#!/usr/bin/env python3
"""Unified CLI for SSL pre-training and linear probing."""
import argparse, torch, torch.optim as optim, tqdm, numpy as np, random, os
from biosignal.data import FolderPerParticipant
from biosignal.augment import BiosignalAugment
from biosignal.ssl import SSLModel, ContrastiveKoLeo
from biosignal.probe import LinearProbe, load_csv
from torch.utils.data import DataLoader

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def train_ssl(args):
    if torch.cuda.is_available():  # NVIDIA GPUs (Linux/Windows)
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # Apple-silicon GPUs (M-series)
        device = torch.device("mps")
    else:  # Fallback: plain CPU
        device = torch.device("cpu")

    print("Using device:", device)
    ds = FolderPerParticipant(args.root, args.seg_len, preprocess=True)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    aug = BiosignalAugment(cutout_p=args.cutout)
    model = SSLModel(ema=args.ema).to(device)
    loss_fn = ContrastiveKoLeo()
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x1, x2 in loader:
            x1 = torch.stack([aug(s) for s in x1]).to(device)
            x2 = torch.stack([aug(s) for s in x2]).to(device)
            z1, z2 = model(x1, x2)
            loss = loss_fn(z1, z2)
            opt.zero_grad(); loss.backward(); opt.step()
            model._update_m()
            losses.append(loss.item())
        print(f"Epoch {epoch:03d}  loss {np.mean(losses):.4f}")
        if (epoch+1)%50==0:
            os.makedirs('ckpts', exist_ok=True)
            torch.save(model.enc.state_dict(), f"ckpts/encoder_{epoch}.pt")

def train_probe(args):
    if torch.cuda.is_available():  # NVIDIA GPUs (Linux/Windows)
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # Apple-silicon GPUs (M-series)
        device = torch.device("mps")
    else:  # Fallback: plain CPU
        device = torch.device("cpu")

    print("Using device:", device)
    xs, ys = load_csv(args.labels)
    probe = LinearProbe(args.encoder_ckpt, num_classes=args.num_classes, task=args.task).to(device)
    opt = optim.AdamW(probe.head.parameters(), lr=1e-2, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss() if args.task=='regression' else torch.nn.CrossEntropyLoss()
    B = args.batch
    for epoch in range(50):
        idx = np.random.permutation(len(xs))
        epoch_loss = 0.
        for i in range(0, len(idx), B):
            subset = idx[i:i+B]
            data = [torch.from_numpy(np.load(xs[j])).float().unsqueeze(0) for j in subset]
            x = torch.stack(data).to(device)
            y = torch.tensor([ys[j] for j in subset]).to(device)
            out = probe(x).squeeze()
            if args.task=='classification':
                y = y.long()
            loss = loss_fn(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()*len(subset)
        print(f"Probe epoch {epoch:02d} loss {epoch_loss/len(xs):.4f}")

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    ssl_p = sub.add_parser('pretrain')
    ssl_p.add_argument('--root', required=True)
    ssl_p.add_argument('--seg_len', type=int, default=60*64)
    ssl_p.add_argument('--batch', type=int, default=256)
    ssl_p.add_argument('--epochs', type=int, default=300)
    ssl_p.add_argument('--lr', type=float, default=1e-3)
    ssl_p.add_argument('--ema', type=float, default=0.99)
    ssl_p.add_argument('--cutout', type=float, default=0.4)
    probe_p = sub.add_parser('probe')
    probe_p.add_argument('--encoder_ckpt', required=True)
    probe_p.add_argument('--labels', required=True)
    probe_p.add_argument('--task', choices=['classification','regression'], default='classification')
    probe_p.add_argument('--num_classes', type=int, default=2)
    probe_p.add_argument('--batch', type=int, default=512)
    args = p.parse_args()
    set_seed()
    if args.cmd=='pretrain':
        train_ssl(args)
    else:
        train_probe(args)

if __name__ == '__main__':
    main()
