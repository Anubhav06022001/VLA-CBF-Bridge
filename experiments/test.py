#!/usr/bin/env python
import os
import sys
import time
import numpy as np
import torch
import mujoco as mj
import glfw
import pickle
from pathlib import Path

from src.control.task_alignment import (compute_task_aligned_P, 
                                        compute_task_direction, 
                                        compute_projection_matrices,
                                        compute_task_deviation)
from src.control.cbf_qp_multi import (cbf_qp_multi, cbf_qp_osqp)
from src.franka.joint_limit_cbf import (joint_limit_barrier, grad_joint_limit_barrier)
from src.vocbf.barrier_net import BarrierNet2
from src.utils.mujoco_ids import initialize_ids
from src.utils.mouse_keyboard import MouseKeyboard
from src.utils.video_recorder import VideoRecorder
from src.safety.safety_function import ell_from_distance
from src.safety.distance_utils import compute_min_dist_from_q, get_robot_collision_geom_ids, get_obstacle_geom_ids
from src.config.franka import (q_min, q_max, simend, q_offsets)

# ===================== CONFIG =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
print("================================" ,PROJECT_ROOT)
xml_path = PROJECT_ROOT / "assets" / "franka_emika_panda" / "scene_cylinder.xml"

# ===== INITIAL POSE TO TEST =====
# q_init_target = np.array([-2.0, 0.8, -1.8, -1.0, 2.5, 1.0, -2.1])
# q_init_target = np.array([-2.6, -0.3, 0.7, -2.4, -2.6, 2.6, -2.4]) 
q_init_target = np.array([-1.2, 1.5, -1.9, -1.9, 2.5, 2.4, 0.2])    # working alpha =800
# # q_init_target = np.array([ 2.122,  0.356,  1.206 ,-3.01 ,  2.723 , 3.121,0]) 

d_safe =0.2
# ================= LOAD NORMALIZATION =================
x_mean = np.load(PROJECT_ROOT / "models" / "dyn_x_mean.npy")
x_std  = np.load(PROJECT_ROOT /  "models" / "dyn_x_std.npy")

# ================= LOAD BARRIER MODEL =================
B_net = BarrierNet2()
model_path = PROJECT_ROOT / "models" / "vocbf.pt" # Update to your latest model name
B_net.load_state_dict(torch.load(model_path, map_location="cpu"))
B_net.eval()

# ================= MUJOCO INIT =================
model = mj.MjModel.from_xml_path(str(xml_path))
data = mj.MjData(model)

robot_geom_ids = get_robot_collision_geom_ids(model)
obstacle_geom_ids = get_obstacle_geom_ids(model)
joint_ids, actuator_ids, site_ids, body_ids = initialize_ids(model)
joint_idx = [joint_ids[f"joint{i}"] for i in range(1,8)]
cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, 'wrist_camera')

# ================= TELEPORT TO INITIAL POSE =================
# 1. Reset physics
mj.mj_resetData(model, data)

# 2. Capture the default "home" pose before we overwrite it (used for the goal calculation)
q_home = data.qpos[joint_idx].copy()

# 3. Teleport joints
data.qpos[joint_idx] = q_init_target
data.qvel[joint_idx] = 0.0

# 4. Pre-fill actuators so the PD controller doesn't snap back to 0
for i in range(7):
    data.ctrl[actuator_ids[f"actuator{i+1}"]] = q_init_target[i]

# 5. Update kinematics for the new pose
mj.mj_forward(model, data)

# ================= CONTROLLER =================
def controller(model, data):
    # ---- current state ----
    q  = data.qpos[joint_idx]
    qd = data.qvel[joint_idx]

    # ================= CBF-QP SETUP =================
    q_des = q_home + q_offsets
    tau_ref = q_des 

    A_list = []
    b_list = []
    alpha = 5000  # Lowered from 5000 for smoother behavior

    # ---------- (A) Joint-limit CBF ----------
    for i in range(7):
        B_i = (q_max[i] - q[i]) * (q[i] - q_min[i])
        grad = np.zeros(7)
        grad[i] = (q_max[i] - q[i]) - (q[i] - q_min[i])

        A_list.append(grad)
        # b_list.append(-alpha * B_i)
        b_val_limit = np.dot(grad, q) - alpha * B_i
        b_list.append(b_val_limit)

    # ---------- (B) Learned V-OCBF ----------
    x = np.concatenate([q, qd])
    x_norm = (x - x_mean) / x_std
    x_torch = torch.tensor(x_norm, dtype=torch.float32, requires_grad=True)

    B_val = B_net(x_torch)

    grad_B_full = torch.autograd.grad(
        outputs=B_val,
        inputs=x_torch,
        grad_outputs=torch.ones_like(B_val)
    )[0].detach().numpy()

    grad_B_q = grad_B_full[:7]

    A_list.append(grad_B_q)
    # b_list.append(-alpha * B_val.item())
    b_val_learned = np.dot(grad_B_q, q) - alpha * B_val.item()
    b_list.append(b_val_learned)
   
    min_dist = compute_min_dist_from_q(model, data, q, robot_geom_ids, obstacle_geom_ids, set_state=False)
    ell_val = ell_from_distance(min_dist, d_safe)

    # ---------- solve QP ----------
    tau_safe = cbf_qp_osqp(tau_ref, A_list, b_list)
    
    print(f"B(x): {B_val.item():.4f} | l: {float(ell_val):.4f} | ||grad_B||: {np.linalg.norm(grad_B_q):.4f} | ncon = {data.ncon}")

    # ---------- apply control ----------
    for i in range(7): 
        data.ctrl[actuator_ids[f"actuator{i+1}"]] = tau_safe[i] 

# ================= GLFW INIT & RENDERING =================
if not glfw.init():
    raise Exception("Could not initialize GLFW")

window = glfw.create_window(1800, 1200, "Demo", None, None)
if not window:
    glfw.terminate()
    raise Exception("Could not create GLFW window")

glfw.make_context_current(window)
glfw.swap_interval(1)

cam = mj.MjvCamera()
opt = mj.MjvOption()
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, int(mj.mjtFontScale.mjFONTSCALE_150.value))

mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)

cam.azimuth = 180.86
cam.elevation = -15.95
cam.distance = 3.22
cam.lookat = np.array([0.0, 0.0, 0.0])

mousekbd = MouseKeyboard(model, data, scene, cam)

glfw.set_key_callback(window, mousekbd.keyboard)
glfw.set_cursor_pos_callback(window, mousekbd.mouse_move)
glfw.set_mouse_button_callback(window, mousekbd.mouse_button)
glfw.set_scroll_callback(window, mousekbd.scroll)

mj.set_mjcb_control(controller)

# Offscreen camera setup for reading pixels
offscreen_cam = mj.MjvCamera()
offscreen_cam.type = mj.mjtCamera.mjCAMERA_FIXED
offscreen_cam.fixedcamid = cam_id

viewport_width, viewport_height = glfw.get_framebuffer_size(window)
rec = VideoRecorder(fps=10)

# ================= SIM LOOP =================
while not glfw.window_should_close(window):
    time_prev = data.time

    while data.time - time_prev < 1.0/60.0:
        mj.mj_step(model, data)

    if data.time >= simend:
        break

     # 1. Render wrist camera to a hidden 128x128 buffer
    mj.mjv_updateScene(model, data, opt, None, offscreen_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    read_vport = mj.MjrRect(0, 0, 128, 128)
    mj.mjr_render(read_vport, scene, context)    

    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    mj.mjv_updateScene(
        model, data, opt, None, cam,
        int(mj.mjtCatBit.mjCAT_ALL.value), scene
    )
    mj.mjr_render(viewport, scene, context)
    rec.capture(viewport, context)

    # Render Inset Wrist View (Visual only, 320x240)
    inset_w, inset_h = 320, 240
    inset_vport = mj.MjrRect(viewport_width - inset_w, viewport_height - inset_h, inset_w, inset_h)
    mj.mjv_updateScene(model, data, opt, None, offscreen_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(inset_vport, scene, context)
    rec.capture(read_vport, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
# rec.save(PROJECT_ROOT / "outputs" / "videos" / "CoRL_env" / "vocbf_single_test.mp4")