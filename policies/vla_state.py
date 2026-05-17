# policies/vla_state.py
import threading
import numpy as np

class SharedVLAState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_image = None
        self.target_xyz = np.array([0.5, 0.0, 0.4]) # Default starting target
        self.is_ready = False

