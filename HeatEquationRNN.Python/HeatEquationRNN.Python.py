#!/usr/bin/env python3
"""heat_rnn.py

Kleines, lauffaehiges Beispiel (PyTorch) fuer ein RNN-basiertes Modell, das die
zeitabhaengige 2D-Waermeleitung vorhersagt.

Enthalten:
- Datengenerator (explizites Finite-Difference-Schema) fuer Trainingsdaten
- Encoder (CNN) -> LSTM -> Decoder (Deconv) Architektur
- optionale Physik-Regularisierung (PDE-Residual) als Loss-Term
- Trainingsloop, Loss-Speicherung, Beispielvorhersagen als PNG

Aufruf:
    python heat_rnn.py --epochs 6 --device cpu

Hinweis: fuer ernsthafte Experimente anpassen (Gittergroesse, Epochen, Batchsize).
"""

import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# -------------------- Hilfsfunktionen / Datengenerator --------------------

def laplacian_np(u, dx, dy):
    """Numerische Laplace-Operator (NumPy) fuer den Datengenerator.
    Randbedingungen: einfache Neumann-Approximation (Spiegeln).
    u: (H,W)
    """
    H, W = u.shape
    lap = np.zeros_like(u)
    lap[1:-1,1:-1] = (
        (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[0:-2,1:-1]) / dx**2
        + (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,0:-2]) / dy**2
    )
    # Kanten (einfaches Spiegeln)
    lap[0,1:-1] = (
        (u[1,1:-1] - 2*u[0,1:-1] + u[1,1:-1]) / dx**2
        + (u[0,2:] - 2*u[0,1:-1] + u[0,0:-2]) / dy**2
    )
    lap[-1,1:-1] = (
        (u[-2,1:-1] - 2*u[-1,1:-1] + u[-2,1:-1]) / dx**2
        + (u[-1,2:] - 2*u[-1,1:-1] + u[-1,0:-2]) / dy**2
    )
    lap[1:-1,0] = (
        (u[2:,0] - 2*u[1:-1,0] + u[0:-2,0]) / dx**2
        + (u[1:-1,1] - 2*u[1:-1,0] + u[1:-1,1]) / dy**2
    )
    lap[1:-1,-1] = (
        (u[2:,-1] - 2*u[1:-1,-1] + u[0:-2,-1]) / dx**2
        + (u[1:-1,-2] - 2*u[1:-1,-1] + u[1:-1,-2]) / dy**2
    )
    lap[0,0] = lap[0,1]
    lap[0,-1] = lap[0,-2]
    lap[-1,0] = lap[-2,0]
    lap[-1,-1] = lap[-2,-1]
    return lap


def step_explicit(u, alpha, dx, dy, dt):
    return u + dt * alpha * laplacian_np(u, dx, dy)


def random_initial_field(H, W, n_gaussians=2):
    x = np.linspace(0,1,H)
    y = np.linspace(0,1,W)
    X, Y = np.meshgrid(x, y, indexing='ij')
    field = np.zeros((H, W))
    for _ in range(n_gaussians):
        cx = np.random.rand()
        cy = np.random.rand()
        amp = 0.5 + np.random.rand() * 0.5
        sigma = 0.04 + 0.12 * np.random.rand()
        field += amp * np.exp(-((X-cx)**2 + (Y-cy)**2) / (2*sigma*sigma))
    return np.clip(field, 0.0, None)


# -------------------- PyTorch Dataset ------------------------------------

class HeatSeqDataset(Dataset):
    def __init__(self, sequences, K=3):
        # sequences: numpy array (N, T, H, W)
        self.seq = torch.tensor(sequences, dtype=torch.float32)
        self.K = K

    def __len__(self):
        return self.seq.shape[0] * (self.seq.shape[1] - 1)

    def __getitem__(self, idx):
        seq_idx = idx // (self.seq.shape[1] - 1)
        t_idx = idx % (self.seq.shape[1] - 1)
        K = self.K
        start = max(0, t_idx - (K-1) + 1)
        window = self.seq[seq_idx, start:t_idx+1]
        if window.shape[0] < K:
            pad = window[0:1].repeat(K - window.shape[0], 1, 1)
            window = torch.cat([pad, window], dim=0)
        # shapes -> window: (K,H,W); target: (H,W)
        return window.unsqueeze(1), self.seq[seq_idx, t_idx+1].unsqueeze(0)


# -------------------- Modell (Encoder -> LSTM -> Decoder) ----------------

class Encoder(nn.Module):
    def __init__(self, latent_dim, H=24, W=24):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 12, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(12, 24, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.ReLU(),
        )
        conv_out_size = 48 * (H // 8) * (W // 8)
        self.fc = nn.Linear(conv_out_size, latent_dim)
    
    def forward(self, x):
        print("Encoder input shape:", x.shape)  # DEBUG
        # sicherstellen, dass Shape (B,1,H,W) ist
        if x.dim() == 3:
            x = x.unsqueeze(1)
        z = self.conv(x)
        z = z.view(z.shape[0], -1)
        return self.fc(z)


class Decoder(nn.Module):
    def __init__(self, latent_dim, H=24, W=24):
        super().__init__()
        self.H = H
        self.W = W
        self.fc = nn.Linear(latent_dim, 48 * (H // 8) * (W // 8))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(12, 1, 4, stride=2, padding=1),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.shape[0], 48, self.H // 8, self.W // 8)
        return self.deconv(x)


class HeatRNN(nn.Module):
    def __init__(self, latent_dim=64, H=24, W=24, K=3):
        super().__init__()
        self.K = K
        self.encoder = Encoder(latent_dim, H=H, W=W)
        self.rnn = nn.LSTM(latent_dim, latent_dim, batch_first=True)
        self.decoder = Decoder(latent_dim, H=H, W=W)

    def forward(self, window):
        # window: (B, K, 1, H, W)
        B, K, C, Hs, Ws = window.shape
        encs = []
        for k in range(K):
            xk = window[:, k, :, :, :]   # (B,1,H,W)
            encs.append(self.encoder(xk))

        encs = torch.stack(encs, dim=1)  # (B,K,latent)
        out_seq, _ = self.rnn(encs)
        last = out_seq[:, -1]
        pred = self.decoder(last)
        return pred


# -------------------- Torch-Laplacian fuer Physik-Loss ---------------------

def torch_laplacian(u, dx, dy):
    # u: (B,1,H,W)
    lap = torch.zeros_like(u)
    lap[:,:,1:-1,1:-1] = (
        (u[:,:,2:,1:-1] - 2*u[:,:,1:-1,1:-1] + u[:,:,0:-2,1:-1]) / dx**2 +
        (u[:,:,1:-1,2:] - 2*u[:,:,1:-1,1:-1] + u[:,:,1:-1,0:-2]) / dy**2
    )
    lap[:,:,0,1:-1] = (
        (u[:,:,1,1:-1] - 2*u[:,:,0,1:-1] + u[:,:,1,1:-1]) / dx**2 +
        (u[:,:,0,2:] - 2*u[:,:,0,1:-1] + u[:,:,0,0:-2]) / dy**2
    )
    lap[:,:,-1,1:-1] = (
        (u[:,:,-2,1:-1] - 2*u[:,:,-1,1:-1] + u[:,:,-2,1:-1]) / dx**2 +
        (u[:,:, -1,2:] - 2*u[:,:, -1,1:-1] + u[:,:, -1,0:-2]) / dy**2
    )
    lap[:,:,1:-1,0] = (
        (u[:,:,2:,0] - 2*u[:,:,1:-1,0] + u[:,:,0:-2,0]) / dx**2 +
        (u[:,:,1:-1,1] - 2*u[:,:,1:-1,0] + u[:,:,1:-1,1]) / dy**2
    )
    lap[:,:,1:-1,-1] = (
        (u[:,:,2:,-1] - 2*u[:,:,1:-1,-1] + u[:,:,0:-2,-1]) / dx**2 +
        (u[:,:,1:-1,-2] - 2*u[:,:,1:-1,-1] + u[:,:,1:-1,-2]) / dy**2
    )
    lap[:,:,0,0] = lap[:,:,0,1]
    lap[:,:,0,-1] = lap[:,:,0,-2]
    lap[:,:,-1,0] = lap[:,:,-2,0]
    lap[:,:,-1,-1] = lap[:,:,-2,-1]
    return lap


# -------------------- Training & Evaluation --------------------------------

def train_and_eval(args):
    # Settings
    device = torch.device(args.device)
    alpha = args.alpha
    H = args.H; W = args.W
    dx = 1.0 / (H - 1); dy = 1.0 / (W - 1)
    dt = args.dt if args.dt is not None else 0.25 * min(dx*dx, dy*dy) / alpha

    # Generate data
    print('Generating data...')
    all_sequences = []
    for s in range(args.n_sequences):
        u0 = random_initial_field(H, W, n_gaussians=args.n_gaussians)
        seq = [u0]
        u = u0.copy()
        for n in range(1, args.sequence_length):
            u = step_explicit(u, alpha, dx, dy, dt)
            seq.append(u.copy())
        all_sequences.append(np.stack(seq, axis=0))
    all_sequences = np.stack(all_sequences, axis=0)
    all_sequences = all_sequences / all_sequences.max()

    n_train = int(args.train_split * all_sequences.shape[0])
    train_data = all_sequences[:n_train]
    val_data   = all_sequences[n_train:]

    train_loader = DataLoader(HeatSeqDataset(train_data, K=args.K), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(HeatSeqDataset(val_data, K=args.K), batch_size=args.batch_size, shuffle=False)

    # Model
    model = HeatRNN(latent_dim=args.latent_dim, H=H, W=W, K=args.K).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs+1):
        model.train()
        t_loss = 0.0; t_count = 0
        for window, target in train_loader:
            window = window.to(device); target = target.to(device)
            pred = model(window)
            loss_data = mse(pred, target)
            loss_phys = torch.tensor(0.0, device=device)
            if args.phys_reg_lambda > 0.0:
                last = window[:, -1]
                ut_approx = (pred - last) / dt
                lap = torch_laplacian(last, dx, dy)
                residual = ut_approx - alpha * lap
                loss_phys = mse(residual, torch.zeros_like(residual))
            loss = loss_data + args.phys_reg_lambda * loss_phys
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            t_loss += loss.item() * window.shape[0]; t_count += window.shape[0]
        train_losses.append(t_loss / t_count)

        # validation
        model.eval()
        v_loss = 0.0; v_count = 0
        with torch.no_grad():
            for window, target in val_loader:
                window = window.to(device); target = target.to(device)
                pred = model(window)
                loss_data = mse(pred, target)
                loss_phys = torch.tensor(0.0, device=device)
                if args.phys_reg_lambda > 0.0:
                    last = window[:, -1]
                    ut_approx = (pred - last) / dt
                    lap = torch_laplacian(last, dx, dy)
                    residual = ut_approx - alpha * lap
                    loss_phys = mse(residual, torch.zeros_like(residual))
                loss = loss_data + args.phys_reg_lambda * loss_phys
                v_loss += loss.item() * window.shape[0]; v_count += window.shape[0]
        val_losses.append(v_loss / v_count)

        print(f"Epoch {epoch}/{args.epochs} | train={train_losses[-1]:.4e} | val={val_losses[-1]:.4e}")

    # Save losses and model
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'args': vars(args)}, os.path.join(out_dir, 'heat_rnn.pth'))
    np.save(os.path.join(out_dir, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(out_dir, 'val_losses.npy'), np.array(val_losses))

    # Plot loss
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.title('Losskurve')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'))
    plt.close()

    # Recursive rollout example: n steps
    model.eval()
    with torch.no_grad():
        sample = torch.tensor(val_data[0:1], dtype=torch.float32).to(device)  # (1,T,H,W)
        K = args.K
        # korrektes Format: (B,K,1,H,W)
        window_frames = sample[:, 0:K].unsqueeze(2)  # (1,K,1,H,W)

        M = min(6, sample.shape[1] - K)
        preds = []

        for m in range(M):
            pred = model(window_frames.to(device))        # (1,1,H,W)
            preds.append(pred.cpu().numpy()[0, 0])       # nur (H,W)

            # altes Fenster verschieben, neue Prediction anhängen
            window_frames = torch.cat([window_frames[:, 1:], pred.unsqueeze(1)], dim=1)

        gt = sample[0, K:K+M].cpu().numpy()  # Ground truth (M, H, W)


    # Save example images
    # Save example images
    for i in range(M):
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        ax[0].imshow(preds[i], origin='lower')
        ax[0].set_title('pred')

        ax[1].imshow(gt[i], origin='lower')  # jetzt gleiche Form (H,W)
        ax[1].set_title('gt')

        ax[2].imshow(np.abs(preds[i] - gt[i]), origin='lower')
        ax[2].set_title('abs error')

        for a in ax:
            a.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'example_step_{i+1}.png'))
        plt.close()

    print(f"Training abgeschlossen. Ergebnisse in: {out_dir}")


# -------------------- CLI / main -----------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu')
    p.add_argument('--H', type=int, default=24)
    p.add_argument('--W', type=int, default=24)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--dt', type=float, default=None)
    p.add_argument('--n_sequences', type=int, default=80)
    p.add_argument('--sequence_length', type=int, default=8)
    p.add_argument('--n_gaussians', type=int, default=2)
    p.add_argument('--train_split', type=float, default=0.8)
    p.add_argument('--K', type=int, default=3)
    p.add_argument('--latent_dim', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--phys_reg_lambda', type=float, default=1.0,
                   help='Weight of physics residual in loss (0 to disable)')
    p.add_argument('--out_dir', type=str, default='./heat_rnn_out')
    p.add_argument('--seed', type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    train_and_eval(args)


if __name__ == '__main__':
    main()

