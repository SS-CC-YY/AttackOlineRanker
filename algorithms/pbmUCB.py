import numpy as np
import sys
import random
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utlis.utlis import is_power2
import time

class PBMUCB(object):
    def __init__(self, K, env,T):
        self.K = K
        self.env = env
        self.T = T

        self.items = self.env.items
        self.L = len(self.items)

        self.S = {}
        self.N_tilde = {}
        self.N = {}
        self.U = {}


        for i in range(self.L):
            self.S[i] = 0
            self.N[i] = 0
            self.N_tilde[i] = 0

        self.rewards = np.zeros(self.T)

    def run(self):
        ## Coefficient for UCB
        d = lambda p, q: p*np.log(1e-10 + p/q) + (1-p)*np.log(1e-10 + (1-p)/(1-q))
        f = lambda t: t*((np.log(t))**3)

        num_t_click = []
        cost = []

        for t in range(self.T):
            for  i in range(self.L):
                if self.N[i] == 0:
                    self.U[i] = float('inf')
                else:
                    self.U[i] = self.S[i]/self.N_tilde[i] + np.sqrt(1.5*self.N[i]/self.N_tilde[i]) * np.sqrt(1.5*np.log(t)/self.N_tilde[i])
                    self.U[i] = min(1,max(self.U[i], 0))
            At = list(dict(sorted(self.U.items(), key=lambda x: x[1], reverse=True)).keys())[:self.K]
            # print(At)
            x, r = self.env.feedback(At)

            self.rewards[t] = r

            for i in range(self.K):
                self.N[At[i]] += 1
                self.N_tilde[At[i]] += self.env.beta[i]
                self.S[At[i]] += x[i]

        return self.rewards, cost, num_t_click
    
    def attack_run(self):
        print("general attack on pbmUCB")

        ## Coefficient for UCB
        d = lambda p, q: p*np.log(1e-10 + p/q) + (1-p)*np.log(1e-10 + (1-p)/(1-q))
        f = lambda t: t*((np.log(t))**3)

        num_t_click = []
        cost = []

        self.env.means[np.argsort(self.env.means)[0]] = 0
        target_arm = random.sample(set(np.argsort(self.env.means)[self.K+1:]),1)[0]

        AuxL = np.argsort(self.env.means)[self.K:1:-1]
        AuxL = np.array(AuxL)
        AuxL = np.insert(AuxL, 0, target_arm)

        # self.env.means = np.zeros(len(self.items))
        # target_arm = 0
        # self.env.means[0] = 0.7
        # self.env.means[1] = 0.65
        # self.env.means[2] = 0.6
        # self.env.means[3] = 0.55
        # self.env.means[4] = 0.5
        # AuxL = [0,1,2,3,4]

        eta_item = np.argsort(self.env.means)[0]
        # print(AuxL)
        # print(self.env.means[AuxL])

        for t in range(self.T):
            for  i in range(self.L):
                if self.N[i] == 0:
                    self.U[i] = float('inf')
                else:
                    self.U[i] = self.S[i]/self.N_tilde[i] + np.sqrt(1.5*self.N[i]/self.N_tilde[i]) * np.sqrt(1.5*np.log(t)/self.N_tilde[i])
                    self.U[i] = min(1,max(self.U[i], 0))

            At = list(dict(sorted(self.U.items(), key=lambda x: x[1], reverse=True)).keys())[:self.K]
            At_hat = At.copy()
            # if t%1000 == 0:
            #     print("self.U:", self.U)
            #     print("At:", At)

            # attack part:
            attack = False
            if list(set(At).difference(set(AuxL))) != []:             
                for k in range(self.K):
                    if At[k] not in AuxL:
                        # set the attractiveness to
                        At_hat[k] = eta_item
                        attack = True
            
            # if t%1000 == 0:
            #     print("At_hat:",At_hat)            
                    
            if attack == True:
                cost.append(1)
            else:
                cost.append(0)

            if target_arm in At:
                if At.index(target_arm) == 0:
                    num_t_click.append(1)
                else:
                    num_t_click.append(0)
            else:
                num_t_click.append(0)

            x, r = self.env.feedback(At_hat)

            self.rewards[t] = r

            for i in range(self.K):
                self.N[At[i]] += 1
                self.N_tilde[At[i]] += self.env.beta[i]
                self.S[At[i]] += x[i]

        return self.rewards, cost, num_t_click