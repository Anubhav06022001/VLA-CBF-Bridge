import numpy as np
from pathlib import Path

np.random.seed(42)

# ---------------- PATH ----------------
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
save_path = PROJECT_ROOT / "data" / "vocbf" / "init_configs_cylinder.npy"

# ---------------- FULL JOINT LIMITS ----------------
q_min_full = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
q_max_full = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

# ---------------- OBSTACLE JOINT STATES ----------------
q_obs1 = np.array([-0.541,  1.111, -0.559, -0.217, -0.702,  0.807, 0.0])
q_obs2 = np.array([ 0.563,  1.676,  0.401, -0.643,  0.373, -0.998, 0.0])
q_obs3 = np.array([-0.507,  1.775, -0.407, -0.698, -0.348, -1.544, 0.0])

obstacles_q = [q_obs1, q_obs2, q_obs3]

# Define how far away (in joint space radians) the arm must spawn from the obstacles
# You can increase this if 1.2 still spawns the robot too close.
SAFE_RADIUS = 5

# ---------------- GENERATE (REJECTION SAMPLING) ----------------
N = 500
safe_samples = []

print("Generating safe initial configurations...")

while len(safe_samples) < N:
    # 1. Generate a random sample across the ENTIRE workspace
    sample = np.random.uniform(q_min_full, q_max_full)
    
    # 2. Check distance to all known obstacle configurations
    is_safe = True
    for q_obs in obstacles_q:
        dist = np.linalg.norm(sample - q_obs)
        if dist < SAFE_RADIUS:
            is_safe = False
            break # Throw it away, it's too close to an obstacle
            
    # 3. Keep it if it passed all checks
    if is_safe:
        safe_samples.append(np.round(sample, 3))

safe_samples = np.array(safe_samples)

# ---------------- SAVE ----------------
save_path.parent.mkdir(parents=True, exist_ok=True)
np.save(save_path, safe_samples)
print("Saved:", save_path)
print("Shape:", safe_samples.shape)
print("Example sample:\n", safe_samples[0])