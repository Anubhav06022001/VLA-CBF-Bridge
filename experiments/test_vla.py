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

xml_path = PROJECT_ROOT / "assets" / "franka_emika_panda" / "scene.xml"

# --- MODULAR IMPORTS ---
from src.utils.mujoco_ids import initialize_ids
from src.utils.mouse_keyboard import MouseKeyboard
from src.config.franka import (q_min, q_max, simend, q_offsets)
from policies.vla_state import SharedVLAState
from policies.vla_thread import start_vla_worker

# ================= VLA THREAD INITIALIZATION =================
# 1. Initialize the "Mailbox"
shared_state = SharedVLAState()

# 2. Start the background VLA Thread
# daemon=True means this thread will automatically die when you close the window
vla_thread = threading.Thread(target=start_vla_worker, args=(shared_state,), daemon=True)
vla_thread.start()

# ================= MUJOCO CONTROLLER =================
def controller(model, data):
    global joint_ids, actuator_ids, site_ids

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
    
    # Clip velocity to ensure smooth, safe movement
    dq = np.clip(dq, -0.05, 0.05)

    q = data.qpos[[joint_ids[f"joint{i}"] for i in range(1,8)]]
    q_des = q + dq

    # --- VOCBF FILTER GOES HERE (If applying Phase 2 safety) ---
    # tau_safe = vocbf_model.filter(q_des, ...)
    # For now, applying direct IK
    tau_safe = q_des 

    for i in range(7):
        data.ctrl[actuator_ids[f"actuator{i+1}"]] = tau_safe[i]
    
# ================= MUJOCO SETUP =================
model = mj.MjModel.from_xml_path(str(xml_path))
data = mj.MjData(model)

joint_ids, actuator_ids, site_ids, body_ids = initialize_ids(model)

# --- Set Initial Target to Current Position ---
# This prevents the robot from snapping to [0,0,0] before the VLA boots up
mj.mj_forward(model, data)
with shared_state.lock:
    shared_state.target_xyz = data.site_xpos[site_ids["ee_site"]].copy()

# ================= GLFW RENDERING SETUP =================
if not glfw.init():
    raise Exception("Could not initialize GLFW")
window = glfw.create_window(1900, 1900, "VLA Integration Demo", None, None)
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

# ---- Callbacks ----
glfw.set_key_callback(window, mousekbd.keyboard)
glfw.set_cursor_pos_callback(window, mousekbd.mouse_move)
glfw.set_mouse_button_callback(window, mousekbd.mouse_button)
glfw.set_scroll_callback(window, mousekbd.scroll)

# --- Set the controller callback ---
mj.set_mjcb_control(controller)

# ================= SIMULATION LOOP =================
while not glfw.window_should_close(window):
    time_prev = data.time
    # Advance simulation until next frame (60Hz)
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
    camera_name = 'robot_camera'
    width = int(1.0 * 640)
    height = int(1.0 * 480)
    loc_x = int(viewport_width - width)
    loc_y = int(viewport_height - height)

    offscreen_viewport = mj.MjrRect(loc_x, loc_y, width, height)
    camera_id = model.camera(camera_name).id
    
    offscreen_cam = mj.MjvCamera()
    offscreen_cam.type = mj.mjtCamera.mjCAMERA_FIXED
    offscreen_cam.fixedcamid = camera_id

    # Update and render the robot camera scene
    mj.mjv_updateScene(model, data, opt, None, offscreen_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(offscreen_viewport, scene, context)

    # 3. EXTRACTION FOR VLA
    # Read the pixels we just rendered so the VLA thread can see them
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    mj.mjr_readPixels(rgb, None, offscreen_viewport, context)
    
    # Write to shared memory (flip vertically because OpenGL reads bottom-to-top)
    with shared_state.lock:
        shared_state.latest_image = np.flipud(rgb)

    # Swap buffers
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()