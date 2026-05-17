import numpy as np
from .base import VLAPolicy

class ToyVLA(VLAPolicy):

    def __init__(self):
        self.q_goal = np.array([0.0 , 1.23 , 0.0, -0.895, 0.0, 2.19, 0.724])
    
    def act(self, language, vision):
        if language == "move to final configuration":
            return self.q_goal
        else:
            raise ValueError("Unknown instruction")
 

