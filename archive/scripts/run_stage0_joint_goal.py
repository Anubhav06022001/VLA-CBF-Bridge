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
from src.utils.mujoco_ids import initialize_ids
from src.utils.mouse_keyboard import MouseKeyboard
from src.config.franka import (q_min, q_max, simend, q_offsets)
# ------------------ Config  ------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))
from vla.toy_vla import ToyVLA
vla = ToyVLA()

xml_path = PROJECT_ROOT / "assets" / "franka_emika_panda" / "scene.xml"

button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

q_home = None
initialized = False

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)



def controller(model, data):
    global site_ids, actuator_ids, joint_ids, body_ids
    global q_home, initialized

    if not initialized:
        q_home = data.qpos[[joint_ids[f"joint{i}"]  for i in range(1,8)]].copy() 
        initialized = True

    q = data.qpos[[joint_ids[f"joint{i}"] for i in range(1,8)]]

    language = "move to final configuration"
    q_offsets = vla.act(language, None)

    q_pick = q_home + q_offsets

    q_des = q_pick.copy()

    data.ctrl[actuator_ids["actuator8"]] = 255

    

    # ------------------ apply control ------------------
    for i in range(7):
        data.ctrl[actuator_ids[f"actuator{i+1}"]] = q_des[i] 

model = mj.MjModel.from_xml_path(str(xml_path))
data = mj.MjData(model)


if not glfw.init():
    raise Exception("Could not initialize GLFW")
window = glfw.create_window(1900, 1900, "Demo", None, None)
if not window:
    glfw.terminate()
    raise Exception("Could not create GLFW window")
glfw.make_context_current(window)
glfw.swap_interval(1)

# --- create Mujoco rend/home/anubhav/Documents/value_guided_cbf/src/config/franka.pyring context ---
cam = mj.MjvCamera()
opt = mj.MjvOption()
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, int(mj.mjtFontScale.mjFONTSCALE_150.value))
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
cam.azimuth =180.86
cam.elevation = -15.95
cam.distance = 3.22
cam.lookat = np.array([0.0, 0.0, 0.0])


joint_ids, actuator_ids, site_ids, body_ids = initialize_ids(model)
site_idx = list(site_ids.values()) 

mousekbd = MouseKeyboard(model, data, scene, cam)

# ---- Callbacks ----
glfw.set_key_callback(window, mousekbd.keyboard)
glfw.set_cursor_pos_callback(window, mousekbd.mouse_move)
glfw.set_mouse_button_callback(window, mousekbd.mouse_button)
glfw.set_scroll_callback(window, mousekbd.scroll)



# --- Set the controller callback ---
mj.set_mjcb_control(controller)

# --- Simulation Loop ---
while not glfw.window_should_close(window):
    time_prev = data.time
    # Advance simulation until next frame
    while data.time - time_prev < 1.0 / 60.0:
        mj.mj_step(model, data)
        # print("cube force:", data.cfrc_ext[body_ids["cube"]])
    if data.time >= simend:
        break

    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, opt, None, cam, int(mj.mjtCatBit.mjCAT_ALL.value), scene)
    mj.mjr_render(viewport, scene, context)
    # ******** inset view (code start) **************************************************************************************
    #Settings for inset view
    camera_name = 'robot_camera'
    # width = 0.8*640
    # height = 0.8*480
    width = 1.0*640
    height = 1.0*480
    loc_x=int(viewport_width - width)
    loc_y=int(viewport_height - height)
    height = int(height)
    width = int(width)

    # 1. Create a rectangular viewport in the upper right corner for example.
    offscreen_viewport = mj.MjrRect(loc_x, loc_y, width, height)

    # 2. Set the camera to the specified view
    camera_id = model.camera(camera_name).id
    offscreen_cam = mj.MjvCamera()
    offscreen_cam.type = mj.mjtCamera.mjCAMERA_FIXED
    offscreen_cam.fixedcamid = camera_id

    #3. Update scene for the off-screen camera
    mj.mjv_updateScene(model, data, opt, None, offscreen_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)

    # 4. Render the scene in the offscreen buffer with mjr_render.
    mj.mjr_render(offscreen_viewport, scene, context)

    # ******** inset view (code end) *************************************************************************************************
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()








