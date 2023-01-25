import numpy as np
import sys
import os
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utlis.utlis import is_power2

class Partition:
    def __init__(self, items, k, m):
        self.items = items
        self.k = k
        self.m = m

class TopRank:
    def __init__(self, K, env, T):
        self.K = K
        self.env = env
        self.T = T

        self.L = len(env.items)
        self.S = np.zeros((self.L, self.L)) # S[i,j] = \sum_t U(t,i,j)
        self.N = np.zeros((self.L, self.L)) # N[i,j] = \sum_t |U(t,i,j)| = \sum_t |C(t,i) - C(t,j)| 1{i,j in the same partition}

        self.G = np.zeros((self.L, self.L), dtype = bool) # G[i,j] = 1 iff i is better than j
        self.partitions = {0:Partition(items=range(self.L), k=0, m=self.K)}

        self.rewards = np.zeros(self.T)

    def _criterion(self, S, N):
        c = 3.43
        return S >= np.sqrt(2 * N * np.log(c * np.sqrt(self.T) * np.sqrt(N)))

    def update(self, t, At, x, r):
        self.rewards[t] = r

        clicks = np.zeros(self.L)
        clicks[np.asarray(At)] = x

        # update S and N
        for c in self.partitions:
            partition = self.partitions[c]
            for i in range(partition.m):
                a = partition.items[i]
                for j in range(i+1, len(partition.items)):
                    b = partition.items[j]
                    x = clicks[a] - clicks[b]
                    self.S[a, b] += x
                    self.N[a, b] += np.abs(x)
                    self.S[b, a] -= x
                    self.N[b, a] += np.abs(x)

        if True:#t % 2000 == 0  or is_power2(t):
            updateG = False
            for c in self.partitions:
                partition = self.partitions[c]
                for i in range(partition.m):
                    a = partition.items[i]
                    for j in range(i+1, len(partition.items)):
                        b = partition.items[j]

                        if self.N[a, b] > 0:
                            if self._criterion(self.S[a, b], self.N[a, b]): # a is better than b
                                self.G[a, b] = True
                                updateG = True
                            elif self._criterion(self.S[b, a], self.N[b, a]):
                                self.G[b, a] = True
                                updateG = True

            if updateG:
                self.partitions = {}
                c = 0
                k = 0
                remain_items = set(np.arange(self.L))
                while k < self.K:
                    bad_items = set(np.flatnonzero(np.sum(self.G[np.asarray(list(remain_items)),:], axis=0)))
                    good_items = remain_items - bad_items
                    self.partitions[c] = Partition(items=list(good_items),k=k,m=min(len(good_items), self.K-k))

                    k += len(good_items)
                    remain_items = remain_items.intersection(bad_items)
                    c += 1
                # print(t, [self.partitions[x].items for x in self.partitions])

    def select(self):
        A = [0] * self.K
        for c in self.partitions:
            partition = self.partitions[c]
            partition.items = np.random.permutation(partition.items)
            A[partition.k:partition.k+partition.m] = partition.items[:partition.m]
        return A

    def run(self):
        target_arm = 100
        num_t_click = []
        for t in range(self.T):
            At = self.select()
            x, r = self.env.feedback(At)
            if target_arm in At:
                if x[At.index(target_arm)] == 1:
                    # print(1)
                    num_t_click.append(1)
                else:
                    num_t_click.append(0)
            else:
                # print(3)
                num_t_click.append(0)
            self.update(t, At, x, r)
        return self.rewards, num_t_click

    def attack_run(self):
        print("general attack on Toprank")
        cost = []
        num_t_click = []
        de = 1

        # self.env.means = np.insert(self.env.means, self.L, 0)
        self.env.means[np.argsort(self.env.means)[0]] = 0
        target_arm = random.sample(set(np.argsort(self.env.means)[self.K+1:]),1)[0]

        AuxL = np.argsort(self.env.means)[self.K:1:-1]
        AuxL = np.array(AuxL)
        AuxL = np.insert(AuxL, 0, target_arm)
        eta_item = np.argsort(self.env.means)[0]
        # print(eta_item)
        # print(AuxL)
        for t in range(self.T):
            attack = False
            At = self.select()
            At_hat = At.copy()
            # if t%1000 == 0:
            #     print(At)

            if list(set(At).difference(set(AuxL))) != []:             
                for k in range(self.K):
                    if At_hat[k] not in AuxL:
                        # set the attractiveness to
                        At_hat[k] = eta_item
                        attack = True
                    
            if attack == True:
                cost.append(1)
            else:
                cost.append(0)

            x, r = self.env.feedback(At_hat)
            if target_arm in At:
                if At.index(target_arm) == 0:
                    num_t_click.append(1)
                else:
                    num_t_click.append(0)
            else:
                num_t_click.append(0)
            self.update(t, At, x, r)
        return cost, num_t_click
    
    def attack_quit_run(self):
        print("attack then quit on Toprank")
        target_arm = random.sample(set(np.argsort(self.env.means)),1)[0]
        Threshold_1 = (4*np.log((self.L ** 2)*self.T))/((self.K/self.L)+(1-np.sqrt(1+8*(self.K/self.L)/4)))
        # print(Threshold_1)
        Threshold = 4000
        num_t_click = []

        cost = []
        for t in range(self.T):
            attack = False
            At = self.select()
            x, r = self.env.feedback(At)
            if target_arm in At:
                if At.index(target_arm) == 0:
                    # print(1)
                    num_t_click.append(1)
                else:
                    num_t_click.append(0)
            else:
                # print(3)
                num_t_click.append(0)

            if t <= Threshold:
                for k in range(self.K):
                    if At[k] == target_arm:
                        if x[k] != 1:
                            x[k] = 1
                            attack = True
                    elif At[k] != target_arm :
                        if x[k] != 0:
                            x[k] = 0
                            attack = True
            if attack == True:
                cost.append(1)
            else:
                cost.append(0)

            self.update(t, At, x, r)
        return cost, num_t_click
