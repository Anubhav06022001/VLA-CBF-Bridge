import pickle
import numpy as np
from pathlib import Path
import os

# ================== PATHS ==================
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent 
DATA_PATH = PROJECT_ROOT / "data/vocbf/transition.pkl"
MODELS_DIR = PROJECT_ROOT / "models"

# ================== LOAD DATA ==================
print(f"Loading dataset from {DATA_PATH}...")
with open(DATA_PATH, "rb") as f:
    dataset = pickle.load(f)

# ================== EXTRACT ARRAYS ==================
q_list = []
qd_list = []
u_list = []

for d in dataset:
    q_list.append(d["q"])
    qd_list.append(d["qd"])
    u_list.append(d["u"])

# Convert to numpy arrays
q_arr = np.array(q_list)
qd_arr = np.array(qd_list)
u_arr = np.array(u_list)

# The state 'x' is the concatenation of 'q' and 'qd'
x_arr = np.concatenate([q_arr, qd_arr], axis=1)

# ================== CALCULATE STATS ==================
x_mean = np.mean(x_arr, axis=0)
x_std  = np.std(x_arr, axis=0)

u_mean = np.mean(u_arr, axis=0)
u_std  = np.std(u_arr, axis=0)

# ================== SAVE ==================
os.makedirs(MODELS_DIR, exist_ok=True)

np.save(MODELS_DIR / "dyn_x_mean.npy", x_mean)
np.save(MODELS_DIR / "dyn_x_std.npy", x_std)
np.save(MODELS_DIR / "dyn_u_mean.npy", u_mean)
np.save(MODELS_DIR / "dyn_u_std.npy", u_std)

print("Normalization parameters calculated and saved successfully!")
print(f"x_mean shape: {x_mean.shape}")
print(f"x_std shape: {x_std.shape}")
print(f"u_mean shape: {u_mean.shape}")
print(f"u_std shape: {u_std.shape}")