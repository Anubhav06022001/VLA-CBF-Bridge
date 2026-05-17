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
import cv2

# ------------------ Config  ------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

from vla.toy_vla import ToyVLA
vla = ToyVLA()
from kinematics.diff_ik import differential_ik
# xml_path = PROJECT_ROOT / "assets" / "franka_emika_panda" / "scene.xml"
xml_path = PROJECT_ROOT / "assets" / "franka_emika_panda" / "scene2.xml"

from env.camera import render_rgb
from vla.clip_vla import CLIPVLA

simend = 10  
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

######################################################################

# ------------------ Helper Functions  ------------------
def initialize_ids(model):
    joint_ids = {}
    actuator_ids = {}
    body_ids = {}
    site_ids = {}

    #--------------------------------
    #  franka joint ids( franka padna = 7 joints)
    #----------------------------------
    arm_joint_names = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"
    ]

    for name in arm_joint_names:
        jid = mj.mj_name2id(model, mjtObj.mjOBJ_JOINT,name)
        if jid == -1:
            raise ValueError(f"joint {name} not found in model")
        joint_ids[name] = jid

    #---------------------------
    # gripper joint (two fingers for given gripper in scene.xml)
    #---------------------------
    gripper_joint_names = ["finger_joint1", "finger_joint2"]

    for name in gripper_joint_names:
        jid = mj.mj_name2id(model, mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            raise ValueError(f"Joint {name} not founf in model")
        joint_ids[name] = jid

    #--------------------------------
    # Actuators(7 @ joints + 1 @ gripper)
    #-----------------------------------
    actuator_names = [
        "actuator1", "actuator2", "actuator3", "actuator4", "actuator5"
        , "actuator6" , "actuator7", "actuator8"
    ]
    for name in actuator_names:
        aid = mj.mj_name2id(model, mjtObj.mjOBJ_ACTUATOR,name)
        if  aid == -1:
            raise ValueError(f"Actuatot {name} not found in model")
        actuator_ids[name] = aid

    #--------------------------------------
    # Sites (for whole-body CBFs)
    #--------------------------------------
    site_names = [
        "link2_site",
        "link3_site",
        "link4_site",
        "link5_site",
        "link6_site",
        "ee_site",
    ]

    for name in site_names:
        sid = mj.mj_name2id(model, mjtObj.mjOBJ_SITE, name)
        if sid == -1:
            raise ValueError(f"Site {name} not found in model")
        site_ids[name] = sid


    # --------------------
    # Bodies
    # --------------------
    body_names = [
        "link0", "link1", "link2", "link3",
        "link4", "link5", "link6", "link7",
        "hand", "left_finger", "right_finger","cube","cube2"
    ]

    for name in body_names:
        bid = mj.mj_name2id(model, mjtObj.mjOBJ_BODY, name)
        if bid != -1:
            body_ids[name] = bid

    # # return joint_ids, actuator_ids, site_ids
    # print("ee_site id:", site_ids["ee_site"])
    # print("model.nsite:", model.nsite)
    # for i in range(model.nsite):
    #     print("site", i, mj.mj_id2name(model, mjtObj.mjOBJ_SITE, i))


    return joint_ids, actuator_ids, site_ids, body_ids  


def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    if not (button_left or button_middle or button_right):
        return

    width, height = glfw.get_window_size(window)
    PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT

    if button_right:
        action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)

################################################################################################

vla = CLIPVLA(device="cpu")

latest_img = None
selected_obj = None


def controller(model, data):
    try:
        global selected_obj, joint_ids, actuator_ids

        q = data.qpos[[joint_ids[f"joint{i}"] for i in range(1,8)]]

        if selected_obj is None:
            return
        
        obj_names = ["cube", "cube2"]
        obj_pos = data.xpos[body_ids[obj_names[selected_obj]]]
    
        dq  = differential_ik(model, data, joint_ids, obj_pos, site_ids, lam = 0.05)
        print("dq==", dq)
        q = data.qpos[[joint_ids[f"joint{i}"] for i in range(1,8)]]
        q_des = q + dq



        for i in range(7):
            data.ctrl[actuator_ids[f"actuator{i+1}"]] = q_des[i]

        print("Reaching object:", selected_obj, "pos:", obj_pos)

    except Exception as e:
        print("Controller error:", e)
        raise
#--------------------------------------------------------------------------

model = mj.MjModel.from_xml_path(str(xml_path))
data = mj.MjData(model)
joint_ids, actuator_ids, site_ids, body_ids = initialize_ids(model)
# obj_pos = data.xpos[body_ids["cube"]]



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

# --- Install GLFW callbacks ---
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

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

        # Ensure scene is updated with the SAME camera
        mj.mjv_updateScene(
            model, data, opt, None,
            offscreen_cam,
            mj.mjtCatBit.mjCAT_ALL.value,
            scene
        )

        latest_img = render_rgb(
            model, data,
            scene=scene,
            context=context,
            cam=offscreen_cam,
            width=320,
            height=240
        )

        # Manual crops (Stage-1 assumption)
        obj1 = latest_img[50:150, 50:150]
        obj2 = latest_img[50:150, 200:300]
        object_images = [obj1, obj2]

        # IMPORTANT: assign to selected_obj
        selected_obj = vla.act(object_images, "reach the cube closer to the robot")
        print("Selected object:", selected_obj)

        # for i, img in enumerate(object_images):
        #     cv2.imshow(f"Object {i}", img[:, :, ::-1])

        # cv2.waitKey(1)


    glfw.swap_buffers(window)
    glfw.poll_events()
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------


glfw.terminate()











