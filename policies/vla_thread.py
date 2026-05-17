# policies/vla_thread.py
import time
import numpy as np
from policies.vla_policy import VLAPolicy

def start_vla_worker(shared_state):
    """
    This function runs continuously in the background.
    It takes the shared_state object to communicate with the MuJoCo thread.
    """
    print("[VLA Thread] Loading model into VRAM (4-bit)...")
    vla = VLAPolicy() # This is your HuggingFace loading script
    print("[VLA Thread] Model loaded and ready.")
    
    with shared_state.lock:
        shared_state.is_ready = True
    
    while True:
        # 1. Safely check if a new image is available
        with shared_state.lock:
            img_copy = shared_state.latest_image.copy() if shared_state.latest_image is not None else None

        if img_copy is not None:
            try:
                # 2. Run the heavy inference (~1.5 seconds)
                delta_xyz = vla.predict_action(img_copy)
                
                # 3. Safely update the global target
                with shared_state.lock:
                    shared_state.target_xyz += np.array(delta_xyz)
                    print(f"[VLA Thread] Delta: {delta_xyz} | New Target: {shared_state.target_xyz}")
            except Exception as e:
                print(f"[VLA Thread] Inference Error: {e}")
                
        # Small sleep to prevent CPU hogging
        time.sleep(0.05)

