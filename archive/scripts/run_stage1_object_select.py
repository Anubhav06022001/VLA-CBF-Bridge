#!/usr/bin/env python
import os
import sys
import pathlib
from pathlib import Path
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

import time
import numpy as np
import torch
import mujoco as mj
from mujoco import mjtObj
import glfw
import cv2
from src.utils.mujoco_ids import initialize_ids
from src.utils.mouse_keyboard import MouseKeyboard
from src.config.franka import (q_min, q_max, simend, q_offsets)
# ------------------ Config  ------------------



from vla.toy_vla import ToyVLA
vla = ToyVLA()

# xml_path = PROJECT_ROOT / "assets" / "franka_emika_panda" / "scene.xml"
xml_path = PROJECT_ROOT / "assets" / "franka_emika_panda" / "scene2.xml"

from env.camera import render_rgb
from vla.clip_vla import CLIPVLA

button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

vla = CLIPVLA(device="cpu")

latest_img = None
selected_obj = None

def controller(model, data):
    global selected_obj, joint_ids, actuator_ids, body_ids, site_ids

    if selected_obj is None:
        return

    obj_names = ["cube", "cube2"]
    obj_pos = data.xpos[body_ids[obj_names[selected_obj]]]

    # simple reaching
    sid = site_ids["ee_site"]

    J_pos = np.zeros((3, model.nv))
    J_rot = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, J_pos, J_rot, sid)

    J = J_pos[:, :7]

    x = data.site_xpos[sid]
    e = obj_pos + np.array([0,0,0.1]) - x

    lam = 0.05
    dq = J.T @ np.linalg.inv(J @ J.T + lam*np.eye(3)) @ e
    dq = np.clip(dq, -0.05, 0.05)

    q = data.qpos[[joint_ids[f"joint{i}"] for i in range(1,8)]]
    q_des = q + dq

    for i in range(7):
        data.ctrl[actuator_ids[f"actuator{i+1}"]] = q_des[i]

model = mj.MjModel.from_xml_path(str(xml_path))
data = mj.MjData(model)
joint_ids, actuator_ids, site_ids, body_ids = initialize_ids(model)

if not glfw.init():
    raise Exception("Could not initialize GLFW")
window = glfw.create_window(1900, 1900, "Demo", None, None)
if not window:
    glfw.terminate()
    raise Exception("Could not create GLFW window")
glfw.make_context_current(window)
glfw.swap_interval(1)

# --- create Mujoco rendering context ---
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

#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
while not glfw.window_should_close(window):
    time_prev = data.time
    while data.time - time_prev < 1.0 / 60.0:
        mj.mj_step(model, data)

    if data.time >= simend:
        break

    # ---------- RENDER ----------
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)


    # ******** inset view (code start) **************************************************************************************
    #Settings for inset view
    camera_name = 'robot_camera'
    width = 0.8*640
    height = 0.8*480

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
    # ---------- VISION (SAFE PLACE) ----------
    if selected_obj is None and int(data.time * 60) % 30 == 0:
        latest_img = render_rgb(
            model, data,
            scene=scene,
            context=context,
            cam=offscreen_cam,
            width=320,
            height=240
        )

        obj1 = latest_img[50:150, 50:150]
        obj2 = latest_img[50:150, 200:300]
        object_images = [obj1, obj2]

        # selected_obj = vla.act(object_images, "reach the red object")
        selected_obj = vla.act(object_images, "reach the cube closer to the robot")
        print("Selected object:", selected_obj)

        for i, img in enumerate(object_images):
            cv2.imshow(f"Object {i}", img[:, :, ::-1])

        print("Selected object:", selected_obj)
        cv2.waitKey(1)

    glfw.swap_buffers(window)
    glfw.poll_events()
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------


glfw.terminate()











