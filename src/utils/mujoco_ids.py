import mujoco as mj
from mujoco import mjtObj
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
        "link2_site","link2_site2",
        "link3_site", "link3_site2",  "link3_site3",
        "link4_site","link4_site2","link4_site3", "link4_site4",
        "link5_site","link5_site2","link5_site3","link5_site4","link5_site5",
        "link6_site", "link6_site2","link6_site3", "link6_site4",
        "link7_site", "link7_site2", "link7_site3", "link7_site4","link7_site5","link7_site6",
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
        "hand", "left_finger", "right_finger","cube"
    ]

    for name in body_names:
        bid = mj.mj_name2id(model, mjtObj.mjOBJ_BODY, name)
        if bid != -1:
            body_ids[name] = bid

    # return joint_ids, actuator_ids, site_ids
    # print("ee_site id:", site_ids["ee_site"])
    # print("model.nsite:", model.nsite)
    # for i in range(model.nsite):
    #     print("site", i, mj.mj_id2name(model, mjtObj.mjOBJ_SITE, i))


    return joint_ids, actuator_ids, site_ids, body_ids  