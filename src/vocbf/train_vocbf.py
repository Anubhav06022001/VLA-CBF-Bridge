import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from barrier_net import BarrierNet2
from src.safety.safety_function import ell_from_distance
from src.learning.expectile import expectile_loss
from pathlib import Path

# ================== PATHS ==================
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent

# ================= CONFIG =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available, running on CPU")


gamma = 0.99
tau = 0.9
lr = 0.0005
epochs = 30
d_safe = 0.1
hidden_dim = 256

# ================= DATASET =================
with open("data/vocbf/transition.pkl", "rb") as f:
    dataset = pickle.load(f)

print(f"Loaded {len(dataset)} transitions.")

# ================= NORMALIZATION =================
x_mean = torch.tensor(np.load("models/dyn_x_mean.npy"), dtype=torch.float32).to(DEVICE)
x_std  = torch.tensor(np.load("models/dyn_x_std.npy"),  dtype=torch.float32).to(DEVICE)
u_mean = torch.tensor(np.load("models/dyn_u_mean.npy"), dtype=torch.float32).to(DEVICE)
u_std  = torch.tensor(np.load("models/dyn_u_std.npy"),  dtype=torch.float32).to(DEVICE)

# ================= BARRIER =================
B_net = BarrierNet2().to(DEVICE)
optimizer = torch.optim.Adam(B_net.parameters(), lr=lr)

# ================= BUILD TENSORS =================
q_all  = torch.tensor(np.stack([d["q"] for d in dataset]),  dtype=torch.float32, device=DEVICE)
qd_all = torch.tensor(np.stack([d["qd"] for d in dataset]), dtype=torch.float32, device=DEVICE)

# Extract Next States directly from dataset (Model-Free)
q_next_all  = torch.tensor(np.stack([d["q_next"] for d in dataset]), dtype=torch.float32, device=DEVICE)
qd_next_all = torch.tensor(np.stack([d["qd_next"] for d in dataset]), dtype=torch.float32, device=DEVICE)

u_all    = torch.tensor(np.stack([d["u"] for d in dataset]),  dtype=torch.float32, device=DEVICE)
dist_all = torch.tensor(np.array([d["min_dist"] for d in dataset]), dtype=torch.float32, device=DEVICE)

# ================= TRAIN =================
batch = 256
N = q_all.shape[0]

for epoch in range(epochs):
    perm = torch.randperm(N)
    losses = []

    for i in range(0, N, batch):
        idx = perm[i:i+batch]

        q = q_all[idx]
        qd = qd_all[idx]
        q_next = q_next_all[idx]
        qd_next = qd_next_all[idx]
        u = u_all[idx]
        d = dist_all[idx]

        # Normalize current state
        x_raw = torch.cat([q, qd], dim=1)
        x = (x_raw - x_mean) / (x_std + 1e-6)
        u = (u - u_mean) / (u_std + 1e-6)
        
        # Normalize next state
        x_next_raw = torch.cat([q_next, qd_next], dim=1)
        x_next = (x_next_raw - x_mean) / (x_std + 1e-6)

        ell = torch.tensor([ell_from_distance(v.item(), d_safe) for v in d], device=DEVICE)

        B = B_net(x).squeeze()
        B = torch.clamp(B, -10, 10)

        with torch.no_grad():
            # FULLY MODEL FREE: Evaluate barrier on the true next state from data
            B_next = B_net(x_next).squeeze()
            B_next = torch.clamp(B_next, -10, 10)

        target = (1-gamma)*ell + gamma*torch.minimum(ell, B_next)

        value_loss = expectile_loss(B, target, tau)

        unsafe = (d < 0)
        safe   = (d > 2*d_safe)

        anchor = 0
        if unsafe.any():
            anchor += 0.05*(B[unsafe]+1).pow(2).mean()
        if safe.any():
            anchor += 0.01*(B[safe]-1).pow(2).mean()

        loss = value_loss + anchor

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(B_net.parameters(), 5.0)
        optimizer.step()

        losses.append(loss.item())

    print(f"Epoch {epoch:03d} | loss {np.mean(losses):.4f}")

# ===================== SAVE =====================
os.makedirs("models", exist_ok=True)
torch.save(B_net.state_dict(), "models/vocbf2.pt")
print("Saved learned V-OCBF to models/vocbf2.pt")