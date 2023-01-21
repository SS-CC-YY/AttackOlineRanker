import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utlis.utlis import is_power2
import time

class CascadeUCB(object):
    def __init__(self, K, env, T):
        super(CascadeUCB, self).__init__()
        self.K = K
        self.env = env
        self.T =T

        self.items = self.env.items
        self.d = len(self.items[0])
        self.hat_alpha = np.zeros(len(self.items))