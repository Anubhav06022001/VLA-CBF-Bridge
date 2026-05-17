

vla-franka/
│
├── README.md
├── requirements.txt
│
├── assets/
│   └── franka_emika_panda/
│       ├── scene.xml
│       ├── franka.xml
│       ├── meshes/
│       └── textures/
|
├── src/
│   ├── config  
│   ├── utils.py      
│
├── perception/
│   ├── owlvit_detector.py     # language → bounding boxes
│   ├── camera_utils.py        # RGB extraction helpers
│
├── policies/
│   ├── pi0_policy.py          # π₀-style policy (core)
│   ├── action_space.py        # definition of action parameterization
|
├── env/
│   ├── franka_env.py
│   ├── camera.py                 # image rendering utils
│   └── __init__.py
│
├── controllers/
│   ├── pd_joint.py
│   ├── osc.py
│   └── __init__.py
│
├── vla/
│   ├── base.py                    # VLA interface (important)
│   ├── toy_vla.py                 # stage 0
│   ├── clip_vla.py                # stage 1-2
│   └── __init__.py
│
├── training/
│   ├── train_toy.py
│   ├── train_clip.py
│   └── __init__.py
│ 
├── kinematics/
|   ├── diff_ik.py
|   ├── fk.py
|
├── scripts/
│   ├── run_stage0_joint_goal.py   ← THIS FILE
│   ├── run_stage1_object_select.py
|   ├── run_pi0_reach.py  
│   └── run_stage2_reach.py
│
├── models/
│   └── (optional checkpoints later)
│
└── requirements.txt




## VLA-Franka

This repo studies hierarchical integration of vision-language-action
policies with low-level robot controllers in MuJoCo.

### Structure
- assets/: MuJoCo models (Franka Panda)
- vla/: high-level VLA policies
- controllers/: torque-level controllers
- env/: gym-wrapped MuJoCo environment

### Stages
1. Language → joint goal
2. Vision + language → object selection
3. Vision + language → reaching



# Stage 0: Language → joint target (toy, no vision)

vla/toy_vla.py--> Returns joint goal.

controllers/pd_joint.py--> Consumes joint goal.

experiments/stage0_joint_goal.py--> Loads scene.xml from root assets/.

# Stage 1: Vision + language → object selection

vla/clip_vla.py

experiments/stage1_object_select.py

# Stage 2: Vision + language → goal → controller
