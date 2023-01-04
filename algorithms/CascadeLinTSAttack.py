import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utlis.utlis import is_power2

class CascadeLinTS:
    # Current version is for fixed item set
    def __init__(self, K, env, T):
        super(CascadeLinTS, self).__init__()
        self.K = K
        self.env = env
        self.T = T

        self.items = self.env.items
        self.d = len(self.items[0])
        
        self.S = np.zeros((self.d, self.d))
        self.b = np.zeros(self.d)
        self.Sinv = np.zeros((self.d, self.d))
        self.theta = np.zeros(self.d)

        self.rewards = np.zeros(self.T)

    def run(self):
        num_t_click = []
        for t in range(self.T):
            if True:#t % 5000 == 0 or is_power2(t):
                self.Sinv = np.linalg.pinv(self.S)
                theta = np.dot(self.Sinv, self.b)
                self.theta = np.random.multivariate_normal(theta, self.Sinv)

            At = np.argsort(np.dot(self.items, self.theta))[:: -1][: self.K]

            # if t == np.argmax(self.env.means):
            #     x, r = self.env.feedback(At)
            #     self.rewards[t] = r
            # else:
            #     x, r = self.env.feedback(At)
            #     self.rewards[t] = 0.1

            x, r = self.env.feedback(At)
            self.rewards[t] = r

            first_click = self.K - 1
            if x.sum() > 0:
                first_click = np.flatnonzero(x)[0]

            target_arm = 100
            
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

            A = self.items[At[: first_click + 1]]
            x = x[: first_click + 1]
            self.S += np.matmul(A.T, A)
            self.b += np.dot(x, A)

        return self.rewards, num_t_click

    def attack_run(self):
        a = 0
        b = 0
        c = 0
        num_t_click = []
        cost = []
        for t in range(self.T):
            if True:#t % 5000 == 0 or is_power2(t):
                self.Sinv = np.linalg.pinv(self.S)
                theta = np.dot(self.Sinv, self.b)
                self.theta = np.random.multivariate_normal(theta, self.Sinv)

            At = np.argsort(np.dot(self.items, self.theta))[:: -1][: self.K]

            # if t == np.argmax(self.env.means):
            #     x, r = self.env.feedback(At)
            #     self.rewards[t] = r
            # else:
            #     x, r = self.env.feedback(At)
            #     self.rewards[t] = 0.1

            x, r = self.env.feedback(At)
            self.rewards[t] = r

            first_click = self.K - 1
            if x.sum() > 0:
                first_click = np.flatnonzero(x)[0]


            target_arm = 100            
                    
            # if t%2000 == 0:
            #     print(x)
            #     print(At)
            #     if target_arm in At:
            #         print(np.where(At == target_arm)[0][0])

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
            
            if x.sum() != self.K:              
                # fixed delta equal to 0.1
                de = 1
                if target_arm not in At: # target arm not got selected in K
                    a += 1
                    cost.append(de)
                    self.rewards[t] -= de
                    first_click = self.K-1
                    x = np.array([0]*self.K) # set all other arm not pulled
                
                elif target_arm in At and (np.where(At == target_arm)[0][0] != 0): # target arm in K but not selected
                    b += 1
                    cost.append(de)
                    self.rewards[t] -= de
                    first_click = np.where(At == target_arm)[0][0]  
                    x[np.where(At == target_arm)[0][0]] = 1
                
                else:
                    c += 1
                    # print(2)
                    cost.append(0)

            A = self.items[At[: first_click + 1]]
            x = x[: first_click + 1]
            self.S += np.matmul(A.T, A)
            self.b += np.dot(x, A)

        print("a:%d b:%d c:%d" %(a,b,c))
        return self.rewards, cost, num_t_click

