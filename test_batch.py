import numpy as np
from random import shuffle
import sys
import random
import time
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import warnings
warnings.filterwarnings("ignore")

class BatchRank(object):
    def __init__(self, K, env, T):
        super(BatchRank, self).__init__()
        self.K = K
        self.env = env
        self.T = T

        self.items = self.env.items
        self.L = len(self.items)
        #display set to hold items in k positions
        self.display = np.zeros(K+1, dtype = int)
        
        #Initialization for clicks and views of documents
        self.C_bl = np.zeros((2*self.K+1, self.T, self.L),dtype=int)
        self.N_bl = np.zeros((2*self.K+1, self.T, self.L),dtype=int)
    
        # Active batches
        self.A = set([1])
        # Highest activate batch number
        self.b_max = 1
        self.I = np.zeros((2*self.K+1, 2),dtype=int)
        # Positions in batch
        self.I[1] = [1,self.K]
    
    def DKL(self,p,q):
        if q == 0 or q == 1:
            if p == q:
                return 0
            else:
                return float('inf')
        if p == 0:
            return (1-p)*np.log((1-p)/(1-q))
        if p == 1:
            return p*np.log(p/q)
        
        return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))


    def UpperBound(self,b,d,nl):
        l = self.l[b]
        c_prob = self.C_bl[b,l,d] / nl
        # get q from [c_prob,1]
        bound = np.log(self.T) + 2*np.log(np.log(self.T))

        q= c_prob
        while nl*self.DKL(c_prob,q) < bound and q < 1:
            q += 0.1
        dkl = self.DKL(c_prob, q-0.1)
        return nl * dkl

    def LowerBound(self,b,d,nl):
        l = self.l[b]
        c_prob = self.C_bl[b,l,d] / nl

        # get q from [c_prob, 1]
        bound = np.log(self.T) + 2*np.log(np.log(self.T))
        q = 0
        while nl*self.DKL(c_prob,q) > bound and q < c_prob:
            q += 0.1
        dkl = self.DKL(c_prob,q)
        return nl*dkl

    def get_key(self,b,l):
        return str(b) + "," +str(int(l))
    def len_bactch(self,b):
        return int(self.I[b,1] - self.I[b,0] + 1)

    def DisplayBatch(self,b,t):
        l = self.l[b]
        bl = self.get_key(b,l)
        # print(b, self.B[bl])
        n_min = min(self.N_bl[b,l,i] for i in self.B[bl])
        len_b = self.len_bactch(b)

        # sort them based on number of times displayed
        least_all = np.argsort(self.N_bl[b,l])
        # print(least_all)
        # get only ones in current batch
        least_viewed = [least_all[i] for i in range(self.L) if least_all[i] in self.B[bl]]
        # print(least_viewed)
        # random positions
        pos_rand = list(range(len_b))
        shuffle(pos_rand)

        # print(least_viewed)

        # Put the items positions to ve displayed
        for k in range(self.I[b,0],self.I[b,1]+1):
            self.display[k] = least_viewed[pos_rand[k-self.I[b,0]]]

    def CollectClicks(self,b,t):
        l = self.l[b]
        bl = self.get_key(b,l)
        n_min = min(self.N_bl[b,l,i] for i in self.B[bl])
        len_b = self.len_bactch(b)

        # click array
        cl = np.zeros(len(self.display))
        At = self.display[self.I[b,0]:self.I[b,1]+1]
        cl[self.I[b,0]:self.I[b,1]+1], cr = self.env.feedback(At)

        for k in range(self.I[b,0], self.I[b,1]+1):
            if self.N_bl[b,l,self.display[k]] == n_min:
                self.C_bl[b,l,self.display[k]] += cl[k]
                self.N_bl[b,l,self.display[k]] += 1

    def UpdateBatch(self,b,t):
        l = self.l[b]
        nl = 16 * pow(2,-1) * np.log(self.T)
        # Upper and Lower bound
        Up = np.zeros(self.L)
        Low = np.zeros(self.L)

        bl = self.get_key(b,l)

        if min(self.N_bl[b,l,i] for i in self.B[bl]) > nl:
            for d in self.B[bl]:
                Up[d] = self.UpperBound(b,d,nl)
                Low[d] = self.LowerBound(b,d,nl)
            
            # sort them based on Lower Bound in descending order
            low_all = np.argsort(Low)[::-1]

            # get only ones in current batch
            bl = self.get_key(b,l)
            low_bound = [low_all[i] for i in range(self.L) if low_all[i] in self.B[bl]]
            len_b = self.len_bactch(b)

            B_plus = set(low_bound[:len_b])
            B_minus = self.B[bl] - B_plus

            # Find a splot at the position with the highest rank
            s = 0
            if len(B_minus) == 0:
                return
            max_u = max(Up[i] for i in B_minus)
            for k in range(len_b):
                if Low[low_bound[k]] > max_u:
                    s = k
            
            if s == 0 and (len(self.B[bl]) > len_b):
                # Next Elimination Stage
                # lower bound of last position in batch
                least_val = Low[low_bound[len_b-1]]

                bl_new = self.get_key(b,l+1)
                self.B[bl_new] = set([d for d in self.B[bl] if Up[d] > least_val])
                self.l[b] += 1
                del self.B[bl]
            
            elif s >0:
                # Split

                # Create two new batches: b_max+1, b_max+2
                self.A = (self.A | set([self.b_max+1, self.b_max+2])) - set([b])

                # Parameters for batch b_max + 1
                self.I[self.b_max + 1] = [self.I[b,0], self.I[b,0] + s - 1]
                bl = self.get_key(self.b_max + 1, 0)

                self.B[bl] = set(low_bound[:len_b])
                self.l[self.b_max+1] = 0
                if len(self.B[bl]) == self.K:
                    print("Done: Top K elements:", self.B[bl])
                    return self.B[bl]
                
                self.I[self.b_max + 2] = [self.I[b,0]+s, self.I[b,1]]
                bl = self.get_key(self.b_max+2,0)
                self.B[bl] = set(low_bound[len_b:])
                self.l[self.b_max+2] = 0

                bl = self.get_key(b,l)
                del self.B[bl]

                self.b_max += 2
                print(self.B)

    def run(self):
        # Map list of elements to numbers for simplicity
        dmap = {l:i for i,l in enumerate(range(len(self.items)))}
        # print(dmap)
        
        # First Batch
        bl = self.get_key(1,0)
        self.B = {bl:set(dmap[i] for i in range(self.L))}
        # Stage
        self.l = np.zeros(2*self.K+1, dtype=int)

        for t in range(self.T):
            for b in self.A:
                self.DisplayBatch(b,t)
            for b in self.A:
                self.CollectClicks(b,t)
            for b in self.A:
                self.UpdateBatch(b,t)

        if self.b_max == 1:
            return 1
        return -1


    def attack_run(self):
        for t in range(self.T):
            cost = []
        return cost