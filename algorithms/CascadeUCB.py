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
        self.T = T

        self.cnt = {}
        self.w_hat = {}
        self.U = {}

        for i in range(self.num_arms):
            self.cnt[i] = 0
            self.w_hat[i] = 0

        self.items = self.env.items

    def run(self):
        ## Coefficient for UCB
        d = lambda p, q: p*np.log(1e-10 + p/q) + (1-p)*np.log(1e-10 + (1-p)/(1-q))
        f = lambda t: t*((np.log(t))**3)

        num_t_click = []
        # Suppose the target arm is the 100th arm
        target_arm = 100
        for t in range(self.T):
            for i in range(self.num_arms):
                if self.cnt[i] == 0:
                    self.U[i] = float('inf')
                else:
                    self.U[i] = self.w_hat[i] + np.sqrt(1.5*np.log(t)/self.cnt[i])
                    self.U[i] = min(1, max(self.U[i], 0))

            At = list(dict(sorted(self.U.items(), key=lambda x: x[1], reverse=True)).keys())[:self.seed_size]

            x, r = self.env.feedback(At)

            # reward of selected 10 arm at time t
            self.rewards[t] = r

            # update statistics
            first_click = self.K - 1 # if none of the arm got clicked, which means none of item interested user
            if x.sum() > 0: # means at least one arm got clicked means user find interested item anc click
                first_click = np.flatnonzero(x)[0] # index of the first click arm in K list

            # calculate whether selected target arm
            if target_arm in At:
                if x[np.where(At == target_arm)[0][0]] == 1:
                    # print(1)
                    num_t_click.append(1)
                else:
                    num_t_click.append(0)
            else:
                # print(3)
                num_t_click.append(0)

            ## if none of the items are clicked dont update reward, else update it. Make sure to verify this
            for i in range(self.seed_size):
                if i <= first_click:
                    arm = At[i]
                    r = self.T[arm]*self.w_hat[arm]
                    if i == first_click:
                        r += 1
                    self.w_hat[arm] = r/(self.cnt[arm]+1)
                    self.cnt[arm] += 1
                
                else:
                    break