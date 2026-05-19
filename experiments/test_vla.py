#!/usr/bin/env python
import os
import sys
import time
import numpy as np
import torch
import mujoco as mj
from mujoco import mjtObj
import glfw
import pathlib
from pathlib import Path
import threading

# ------------------ Config & Paths ------------------
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

xml_path = PROJECT_ROOT / "assets" / "franka_emika_panda" / "scene_cylinder.xml"

# --- MODULAR IMPORTS ---
from src.utils.mujoco_ids import initialize_ids
from src.utils.mouse_keyboard import MouseKeyboard
from src.config.franka import (q_min, q_max, simend, q_offsets)

# --- VOCBF SAFETY IMPORTS ---
from src.control.cbf_qp_multi import cbf_qp_osqp
from src.vocbf.barrier_net import BarrierNet2

# --- VLA THREADING IMPORTS ---
from policies.vla_state import SharedVLAState
from policies.vla_thread import start_vla_worker

# ================= VLA THREAD INITIALIZATION =================
shared_state = SharedVLAState()
vla_thread = threading.Thread(target=start_vla_worker, args=(shared_state,), daemon=True)
vla_thread.start()

# ================= LOAD VOCBF MODEL =================
x_mean = np.load(PROJECT_ROOT / "models" / "dyn_x_mean.npy")
x_std  = np.load(PROJECT_ROOT / "models" / "dyn_x_std.npy")

B_net = BarrierNet2()
model_path = PROJECT_ROOT / "models" / "vocbf.pt" 
B_net.load_state_dict(torch.load(model_path, map_location="cpu"))
B_net.eval()

# ================= MUJOCO SETUP =================
model = mj.MjModel.from_xml_path(str(xml_path))
data = mj.MjData(model)

joint_ids, actuator_ids, site_ids, body_ids = initialize_ids(model)
joint_idx = [joint_ids[f"joint{i}"] for i in range(1,8)]

# Set Initial Target to Current Position to prevent snapping
mj.mj_forward(model, data)
with shared_state.lock:
    shared_state.target_xyz = data.site_xpos[site_ids["ee_site"]].copy()

# ================= MUJOCO CONTROLLER (500 Hz) =================
def controller(model, data):
    global joint_ids, actuator_ids, site_ids, joint_idx

    q  = data.qpos[joint_idx]
    qd = data.qvel[joint_idx]

    # 1. Get current EE pos
    sid = site_ids["ee_site"]
    ee_pos = data.site_xpos[sid]

    # 2. Read the VLA target from the Mailbox SAFELY
    with shared_state.lock:
        x_des = shared_state.target_xyz.copy()

    # 3. Differential IK (Moving EE to VLA Target)
    J_pos = np.zeros((3, model.nv))
    J_rot = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, J_pos, J_rot, sid)

    J = J_pos[:, :7]
    e = x_des - ee_pos

    # Damped least squares for IK
    lam = 0.05
    dq = J.T @ np.linalg.inv(J @ J.T + lam*np.eye(3)) @ e
    dq = np.clip(dq, -0.05, 0.05) # Limit max velocity

    # Nominal Target (What the VLA wants)
    q_des = q + dq
    tau_ref = q_des 

    # --- 4. VOCBF FILTER (Safety check) ---
    A_list = []
    b_list = []
    alpha = 800  # Configured barrier aggression

    # A. Joint-limit CBF (Don't break the robot's physical joints)
    for i in range(7):
        B_i = (q_max[i] - q[i]) * (q[i] - q_min[i])
        grad = np.zeros(7)
        grad[i] = (q_max[i] - q[i]) - (q[i] - q_min[i])
        A_list.append(grad)
        b_list.append(np.dot(grad, q) - alpha * B_i)

    # B. Learned V-OCBF (Don't hit the cylinder)
    x = np.concatenate([q, qd])
    x_norm = (x - x_mean) / x_std
    x_torch = torch.tensor(x_norm, dtype=torch.float32, requires_grad=True)

    B_val = B_net(x_torch)
    grad_B_full = torch.autograd.grad(
        outputs=B_val, inputs=x_torch, grad_outputs=torch.ones_like(B_val)
    )[0].detach().numpy()

    grad_B_q = grad_B_full[:7]
    A_list.append(grad_B_q)
    b_list.append(np.dot(grad_B_q, q) - alpha * B_val.item())
   
    # C. Solve QP for Safe Output
    tau_safe = cbf_qp_osqp(tau_ref, A_list, b_list)

    # 5. Send safe commands to Actuators
    for i in range(7): 
        data.ctrl[actuator_ids[f"actuator{i+1}"]] = tau_safe[i] 

# ================= GLFW RENDERING SETUP =================
if not glfw.init():
    raise Exception("Could not initialize GLFW")
window = glfw.create_window(1900, 1900, "VLA Integration Demo", None, None)
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

mousekbd = MouseKeyboard(model, data, scene, cam)
glfw.set_key_callback(window, mousekbd.keyboard)
glfw.set_cursor_pos_callback(window, mousekbd.mouse_move)
glfw.set_mouse_button_callback(window, mousekbd.mouse_button)
glfw.set_scroll_callback(window, mousekbd.scroll)

mj.set_mjcb_control(controller)

# ================= SIMULATION LOOP =================
while not glfw.window_should_close(window):
    time_prev = data.time
    while data.time - time_prev < 1.0 / 60.0:
        mj.mj_step(model, data)

    if data.time >= simend:
        break

    # 1. Main Viewport Rendering
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, opt, None, cam, int(mj.mjtCatBit.mjCAT_ALL.value), scene)
    mj.mjr_render(viewport, scene, context)

    # 2. Inset View (Robot Camera) Rendering
    camera_name = 'wrist_camera'
    width = int(1.0 * 640)
    height = int(1.0 * 480)
    loc_x = int(viewport_width - width)
    loc_y = int(viewport_height - height)

    offscreen_viewport = mj.MjrRect(loc_x, loc_y, width, height)
    camera_id = model.camera(camera_name).id
    
    offscreen_cam = mj.MjvCamera()
    offscreen_cam.type = mj.mjtCamera.mjCAMERA_FIXED
    offscreen_cam.fixedcamid = camera_id

    mj.mjv_updateScene(model, data, opt, None, offscreen_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(offscreen_viewport, scene, context)

    # 3. EXTRACTION FOR VLA
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    mj.mjr_readPixels(rgb, None, offscreen_viewport, context)
    
    with shared_state.lock:
        shared_state.latest_image = np.flipud(rgb)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()