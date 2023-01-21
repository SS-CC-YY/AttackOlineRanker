import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utlis.utlis import is_power2
import time

class CascadeLinUCB(object):
    # Current version is for fixed item set !!!!
    def __init__(self, K, env, T):
        super(CascadeLinUCB, self).__init__()
        self.K = K
        self.env = env
        self.T = T

        self.items = self.env.items
        self.d = len(self.items[0])
        self.beta = np.sqrt(self.d * np.log((1 + self.T/self.d) + 2*np.log(4*self.T))) + 1

        # L = len(self.items)
        self.S = np.zeros((self.d, self.d)) # sigma
        self.b = np.zeros(self.d)
        self.Sinv = np.zeros((self.d, self.d))
        self.theta = np.zeros(self.d)

        self.rewards = np.zeros(self.T)

    def run(self):
        num_t_click = []
        # Suppose the target arm is the 100th arm
        target_arm = 100
        for t in range(self.T):
            if True:#t % 5000 == 0 or is_power2(t):
                # Pseudo-inverse matrix
                self.Sinv = np.linalg.pinv(self.S) 
                self.theta = np.dot(self.Sinv, self.b)


            # sort from large to small
            At = np.argsort(np.dot(self.items, self.theta) + self.beta * (np.matmul(self.items, self.Sinv) * self.items).sum(axis = 1))[:: -1][: self.K]
            # UCB upper boundry 
            # np.dot(self.items, self.theta) + self.beta * (np.matmul(self.items, self.Sinv) * self.items).sum(axis = 1)[::-1] 

            # if np.linalg.cond(self.S) < 1 / sys.float_info.epsilon:
            # 	Sinv = np.linalg.inv(self.S)
            # 	theta = np.dot(Sinv, self.b)

            # 	At = np.argsort(np.dot(self.items, theta) + self.beta * (np.matmul(self.items, Sinv) * self.items).sum(axis = 1))[:: -1][: self.K]
            # else:
            # 	At = np.random.permutation(len(self.items))[: self.K]

            # if t == np.argmax(self.env.means):
            #     x, r = self.env.feedback(At)
            #     self.rewards[t] = r
            # else:
            #     x, r = self.env.feedback(At)
            #     self.rewards[t] = 0.1

            # reward is evaluated by expected cumulative regret
            x, r = self.env.feedback(At)
            # x would be whether click of each arm(binomial distribution)

            # reward of selected 10 arm at time t
            self.rewards[t] = r
            
            if t == 99999:
                print(np.argsort(np.dot(self.items, self.theta) + self.beta * (np.matmul(self.items, self.Sinv) * self.items).sum(axis = 1))[:: 1][: self.K-1])

            # update statistics
            first_click = self.K - 1 # if none of the arm got clicked, which means none of item interested user
            if x.sum() > 0: # means at least one arm got clicked means user find interested item anc click
                first_click = np.flatnonzero(x)[0] # index of the first click arm in K list

            # attacker part:
            # tried attack based on target arm.
            # print((target_arm in At))

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
            
            # A is the clicked item
            A = self.items[At[: first_click + 1]]
            x = x[: first_click + 1]
            self.S += np.matmul(A.T, A) # np.matmul() -> adaptive matrix product
            self.b += np.dot(x, A)

        return self.rewards, num_t_click

    def attack_run(self):
        cost = []
        num_t_click = []
        # Suppose the target arm is the 100th arm
        target_arm = 100
        self.Sinv = np.linalg.pinv(self.S) 
        self.theta = np.dot(self.Sinv, self.b)
        # Total items set
        D_set = np.argsort(np.dot(self.items, self.theta) + self.beta * (np.matmul(self.items, self.Sinv) * self.items).sum(axis = 1))

        # The ordered list with auxiliary items
        AuxL = np.argsort(np.dot(self.items, self.theta) + self.beta * (np.matmul(self.items, self.Sinv) * self.items).sum(axis = 1))[:: 1][: self.K-1]
        # AuxL = [860,300,498,246,628,910,637,750,968]
        AuxL = np.array(AuxL)
        AuxL = np.insert(AuxL, 0, target_arm)
        print(AuxL)
        
        de = 1
        for t in range(self.T):
            if True:#t % 5000 == 0 or is_power2(t):
                # Pseudo-inverse matrix
                self.Sinv = np.linalg.pinv(self.S) 
                self.theta = np.dot(self.Sinv, self.b)

            # sort from large to small
            # At = np.argsort(np.dot(self.items, self.theta) + self.beta * (np.matmul(self.items, self.Sinv) * self.items).sum(axis = 1))[:: -1][: self.K]
            DT = np.argsort(np.dot(self.items, self.theta) + self.beta * (np.matmul(self.items, self.Sinv) * self.items).sum(axis = 1))[:: -1]
            At = DT[:self.K]
            # reward is evaluated by expected cumulative regret
            x, r = self.env.feedback(At)
            dx,dr = self.env.feedback(DT)
    
            # x would be whether click of each arm(binomial distribution)
            # reward of selected 10 arm at time t
            self.rewards[t] = r

            # attacker part:
            # tried attack based on target arm.
            # calculate whether selected target arm
            if target_arm in At:
                if dx[np.where(At == target_arm)[0][0]] == 1:
                    num_t_click.append(1)
                else:
                    num_t_click.append(0)
            else:
                num_t_click.append(0)

            # if t % 1000 == 0:
            #     print(At)

            # Check if the intersecion of recommend list and auxiliary list empty 
            if list(set(At).difference(set(AuxL))) != []:              
                for k in range(len(self.items)):
                    if D_set[k] not in AuxL and D_set[k] in At:
                        # set the item not be clicked
                        cost.append(1)
                        dx[np.where(DT == D_set[k])[0][0]] = 0
                    else:
                        # not attack
                        cost.append(0)
            else:
                # not attack
                cost.append(0)

            # update statistics
            first_click = len(dx) - 1 # if none of the arm got clicked, which means none of item interested user
            if dx.sum() > 0: # means at least one arm got clicked means user find interested item anc click
                first_click = np.flatnonzero(dx)[0] # index of the first click arm in K list

            # A is the clicked item
            A = self.items[DT[: first_click + 1]]
            dx = dx[: first_click + 1]
            self.S += np.matmul(A.T, A) # np.matmul() -> adaptive matrix product
            self.b += np.dot(dx, A)
        return self.rewards, cost, num_t_click