import numpy as np



# ---------------- Simulation ----------------
simend = 20.0

# ---------------- CBF weights ----------------
USE_ANALYTIC_DISTANCE = True
DISTANCE_WEIGHT = 0.3
ALPHA = 2.0

# ---------------- Joint limits ----------------
q_min = np.array([
    -2.8973,
    -1.7628,
    -2.8973,
    -3.0718,
    -2.8973,
    -0.0175,
    -2.8973,
])

q_max = np.array([
     2.8973,
     1.7628,
     2.8973,
    -0.0698,
     2.8973,
     3.7525,
     2.8973,
])

# ---------------- Obstacles ----------------
OBSTACLES = [
    {"p": np.array([0.8, 0.0, 0.40]), "r": 0.12},
    {"p": np.array([0.6, 0.0, 0.40]), "r": 0.12},
]

# ---------------- Link sites ----------------
LINK_SITES = [
    "link2_site",
    "link3_site",
    "link4_site",
    "link5_site",
    "link6_site",
    "ee_site",
]

# ---------------- Nominal pose offsets ---------------- 
q_offsets = np.array([0.0,1.23,0.0,-0.895,0.0,2.19,0.724])    # desired pos [0.8 , 0, 0.03]
# q_offsets = np.array([ 0, 1.52, 0, -0.819,0, -2.03, 0])    # desired pos [0.8, 0, 0.2]

# q_offsets = np.array([0,  0.564, 0, -1.23 ,0,2.34, 0.724])   # desired pos [0.8 , 0, 0.35]