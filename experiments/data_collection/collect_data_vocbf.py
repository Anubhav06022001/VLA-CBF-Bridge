#---------------------------------------------------------
# # FOR RUNNING ON GPU-->export MUJOCO_GL=egl
import mujoco as mj
import numpy as np
import pickle
import pandas as pd
from mujoco import mjtObj
from pathlib import Path
from src.utils.mujoco_ids import initialize_ids
from src.safety.distance_utils import (
    compute_min_dist_from_q,
    get_robot_collision_geom_ids,
    get_obstacle_geom_ids
)
# ------------------ Paths ------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# print("project_root------>", PROJECT_ROOT)
xml_path = PROJECT_ROOT / "assets" / "franka_emika_panda" / "scene_cylinder.xml"
# ------------------ Load model ------------------
model = mj.MjModel.from_xml_path(str(xml_path))
data = mj.MjData(model)
robot_geom_ids = get_robot_collision_geom_ids(model)
obstacle_geom_ids = get_obstacle_geom_ids(model)

joint_ids, actuator_ids, site_ids, body_ids = initialize_ids(model)
joint_idx = [joint_ids[f"joint{i}"] for i in range(1, 8)]
# ------------------ Data buffer ------------------
dataset = []
# ----------------- PD parameters ------------------
kp = 1.0
noise_std = 0
max_steps = 1200
num_episodes = 300
q_min = np.array([ -2.8973, -1.7628,-2.8973,-3.0718,-2.8973,-0.0175, -2.8973,])

q_max = np.array([2.8973,1.7628,.8973,-0.0698, 2.8973,3.7525,2.8973,])

print("Starting data collection...")

for episode in range(num_episodes):

    mj.mj_resetData(model, data)
    data.qpos[joint_idx] = np.random.uniform(q_min, q_max)      # initialise to random intial config
    data.qvel[joint_idx] = 0.1 * np.random.randn(7)
    mj.mj_forward(model, data)
    q_goal = np.random.uniform(q_min, q_max)


    for t in range(max_steps):
        q = data.qpos[joint_idx].copy()
        qd = data.qvel[joint_idx].copy()
        
        data.ctrl[:7] = q_goal
        mj.mj_step(model, data)
        
        q_next = data.qpos[joint_idx].copy()
        qd_next = data.qvel[joint_idx].copy()

        # Calculate distance DURING collection
        min_dist = compute_min_dist_from_q(
            model, data, q, robot_geom_ids, obstacle_geom_ids
        )

        dataset.append({
            "q": q,
            "qd": qd,
            "u": q_goal,
            "q_next": q_next,
            "qd_next": qd_next,
            "min_dist": min_dist  
        })


print("Total samples:", len(dataset))

# ------------------ Save ------------------
PROJECT_ROOT.joinpath("data/vocbf").mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / "data/vocbf/transition.pkl", "wb") as f:
    pickle.dump(dataset, f)

df = pd.DataFrame(dataset)
df.to_csv(PROJECT_ROOT / "data/vocbf/transitions.csv", index=False)

print("Dataset saved.")




