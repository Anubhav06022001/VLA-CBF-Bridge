#!/usr/bin/env python

import os
import sys
import time
import numpy as np
import torch
import mujoco as mj
from mujoco import mjtObj
import glfw



# ------------------ Config  ------------------
xml_path   = "/home/anubhav/Documents/dm_control/dm_control/suite/franka_emika_panda/scene.xml"
# from pathlib import Path
# THIS_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = THIS_DIR.parent

# xml_path = PROJECT_ROOT / "assets" / "franka_emika_panda" / "scene.xml"


simend = 10  
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

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
    # Site id(one site add @ EE)
    #--------------------------------------
    site_ids["ee_site"] = mj.mj_name2id(model, mjtObj.mjOBJ_SITE, "ee_site")

    # --------------------
    # Bodies
    # --------------------
    body_names = [
        "link0", "link1", "link2", "link3",
        "link4", "link5", "link6", "link7",
        "hand", "left_finger", "right_finger","cube"
    ]

    for name in body_names:
        bid = mj.mj_name2id(model, mjtObj.mjOBJ_BODY, name)
        if bid != -1:
            body_ids[name] = bid

    # return joint_ids, actuator_ids, site_ids

    return joint_ids, actuator_ids, site_ids, body_ids  

#####################################################################################
q_home = None
initialized = False
# ---- globals ----
grasp_phase = 0
t_phase = 0.0

def controller(model, data):
    global q_home, initialized
    global grasp_phase, t_phase

    if not initialized:
        q_home = data.qpos[[joint_ids[f"joint{i}"] for i in range(1,8)]].copy()
        initialized = True
        grasp_phase = 0
        t_phase = data.time

    # current joint state
    q = data.qpos[[joint_ids[f"joint{i}"] for i in range(1,8)]]

    # -------------------------
    # Desired posture baseline
    # -------------------------
    q_offsets = np.array([
        0.0,
        1.23,
        0.0,
        -0.895,
        0.0,
        2.19,
        0.724,
    ])

    q_pick = q_home + q_offsets
    q_des = q_pick.copy()

    # -------------------------
    # Phase logic
    # -------------------------

    # Phase 0: open gripper, move to pick pose
    if grasp_phase == 0:
        data.ctrl[actuator_ids["actuator8"]] = 255  # open

        if data.time - t_phase > 1.0:
            grasp_phase = 1
            t_phase = data.time

    # Phase 1: close gripper (
    elif grasp_phase == 1:
        data.ctrl[actuator_ids["actuator8"]] = 0  # close

        if data.time - t_phase > 0.5:
            grasp_phase = 2
            t_phase = data.time

    # Phase 2: preload (tiny upward motion)
    elif grasp_phase == 2:
        data.ctrl[actuator_ids["actuator8"]] = 0
        q_des[1] += -0.05  # VERY small lift

        if data.time - t_phase > 0.5:
            grasp_phase = 3
            t_phase = data.time

    # Phase 3: lift fully
    elif grasp_phase == 3:
        data.ctrl[actuator_ids["actuator8"]] = 0
        q_des[1] += -0.3   

    # -------------------------
    # Send arm commands
    # -------------------------
    for i in range(7):
        data.ctrl[actuator_ids[f"actuator{i+1}"]] = q_des[i]





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


dirname = os.path.dirname(__file__)
xml_path = os.path.join(dirname, xml_path)


model = mj.MjModel.from_xml_path(str(xml_path))
data = mj.MjData(model)
# joint_ids, actuator_ids, site_ids = initialize_ids(model)
joint_ids, actuator_ids, site_ids, body_ids = initialize_ids(model)

if not glfw.init():
    raise Exception("Could not initialize GLFW")
window = glfw.create_window(1200, 900, "Demo", None, None)
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

# --- Simulation Loop ---
while not glfw.window_should_close(window):
    time_prev = data.time
    # Advance simulation until next frame
    while data.time - time_prev < 1.0 / 60.0:
        mj.mj_step(model, data)
        print("cube force:", data.cfrc_ext[body_ids["cube"]])
    if data.time >= simend:
        break

    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, opt, None, cam, int(mj.mjtCatBit.mjCAT_ALL.value), scene)
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()

