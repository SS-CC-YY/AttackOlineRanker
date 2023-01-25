import numpy as np
import sys
import random
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
        
        self.items = self.env.items
        self.L = len(self.items)

        self.cnt = {}
        self.w_hat = {}
        self.U = {}

        for i in range(self.L):
            self.cnt[i] = 0
            self.w_hat[i] = 0

        self.rewards = np.zeros(self.T)

    def attack_run(self):
        print("general attack on cascadeUCB")

        ## Coefficient for UCB
        d = lambda p, q: p*np.log(1e-10 + p/q) + (1-p)*np.log(1e-10 + (1-p)/(1-q))
        f = lambda t: t*((np.log(t))**3)

        # Suppose the target arm is the 10th arm

        cost = []
        num_t_click = []

        self.env.means[np.argsort(self.env.means)[0]] = 0
        target_arm = random.sample(set(np.argsort(self.env.means)[self.K+1:]),1)[0]
        AuxL = np.argsort(self.env.means)[self.K:1:-1]
        AuxL = np.array(AuxL)
        AuxL = np.insert(AuxL, 0, target_arm)

        # self.env.means = np.zeros(len(self.items))
        # target_arm = 0
        # self.env.means[0] = 0.25
        # self.env.means[1] = 0.21
        # self.env.means[2] = 0.209
        # self.env.means[3] = 0.208
        # self.env.means[4] = 0.207
        # AuxL = [0,1,2,3,4]

        eta_item = np.argsort(self.env.means)[0]
        # print("target arm:",target_arm)
        # print(eta_item)
        # print(AuxL)

        for t in range(self.T):
            for i in range(self.L):
                if self.cnt[i] == 0:
                    self.U[i] = float('inf')
                else:
                    self.U[i] = self.w_hat[i] + np.sqrt(1.5*np.log(t)/self.cnt[i])
                    self.U[i] = min(1, max(self.U[i], 0))

            At = list(dict(sorted(self.U.items(), key=lambda x: x[1], reverse=True)).keys())[:self.K]
            # At_hat = list(dict(sorted(self.U.items(), key=lambda x: x[1], reverse=True)).keys())[:self.K]
            At_hat = At.copy()
            # if t%100 == 0:
            #     # print("self.U:", self.U)
            #     print("At:", At)

            # attack part:
            attack = False
            if list(set(At).difference(set(AuxL))) != []:             
                for k in range(self.K):
                    if At_hat[k] not in AuxL:
                        # set the attractiveness to
                        At_hat[k] = eta_item
                        attack = True
            
            # if t%1000 == 0:
            #     # print("self.U:", self.U)
            #     print("At attack:", At)            
                    
            if attack == True:
                cost.append(1)
            else:
                cost.append(0)

            x, r = self.env.feedback(At_hat)
            # if t%1000 == 0:
            #     # print("self.U:", self.U)
            #     print("At_hat:", At_hat)
            #     print("At:",At)
            #     print("x", x)
            #     print()

            # calculate whether selected target arm
            if target_arm in At:
                if At.index(target_arm) == 0:
                    # print(1)
                    num_t_click.append(1)
                else:
                    num_t_click.append(0)
            else:
                num_t_click.append(0)


            # reward of selected 10 arm at time t
            self.rewards[t] = r

            # update statistics
            first_click = self.K # if none of the arm got clicked, which means none of item interested user
            if x.sum() > 0: # means at least one arm got clicked means user find interested item anc click
                first_click = np.flatnonzero(x)[0] # index of the first click arm in K list

            ## if none of the items are clicked dont update reward, else update it. Make sure to verify this
            for i in range(self.K):
                if i <= first_click:
                    arm = At[i]
                    r = self.cnt[arm] * self.w_hat[arm]
                    if i == first_click:
                        # print(arm)
                        r += 1
                    self.w_hat[arm] = r/(self.cnt[arm]+1)
                    self.cnt[arm] += 1
                else:
                    break
        return self.rewards, cost, num_t_click

    def run(self):
        ## Coefficient for UCB
        d = lambda p, q: p*np.log(1e-10 + p/q) + (1-p)*np.log(1e-10 + (1-p)/(1-q))
        f = lambda t: t*((np.log(t))**3)

        # Suppose the target arm is the 10th arm
        target_arm = random.sample(set(np.argsort(self.env.means)),1)[0]
        # target_arm = 0

        num_t_click = []

        for t in range(self.T):
            for i in range(self.L):
                if self.cnt[i] == 0:
                    self.U[i] = float('inf')
                else:
                    self.U[i] = self.w_hat[i] + np.sqrt(1.5*np.log(t)/self.cnt[i])
                    self.U[i] = min(1, max(self.U[i], 0))

            At = list(dict(sorted(self.U.items(), key=lambda x: x[1], reverse=True)).keys())[:self.K]

            x, r = self.env.feedback(At)

            # if t % 1000 == 0:
            #     print(At)
            #     print(r)

            # reward of selected 10 arm at time t
            self.rewards[t] = r

            # update statistics
            first_click = self.K - 1 # if none of the arm got clicked, which means none of item interested user
            if x.sum() > 0: # means at least one arm got clicked means user find interested item anc click
                first_click = np.flatnonzero(x)[0] # index of the first click arm in K list

            # calculate whether selected target arm
            if target_arm in At:
                if At.index(target_arm) == 0:
                    # print(1)
                    num_t_click.append(1)
                else:
                    num_t_click.append(0)
            else:
                # print(3)
                num_t_click.append(0)

            ## if none of the items are clicked dont update reward, else update it. Make sure to verify this
            for i in range(self.K):
                if i <= first_click:
                    arm = At[i]
                    r = self.cnt[arm] * self.w_hat[arm]
                    if i == first_click:
                        r += 1
                    self.w_hat[arm] = r/(self.cnt[arm]+1)
                    self.cnt[arm] += 1
                
                else:
                    break
        return self.rewards, 0, num_t_click