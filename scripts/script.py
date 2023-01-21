# # -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import random
import time
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from utlis.argparsers.simulationargparser import SimulationArgumentParser
from algorithms.TopRank import TopRank
from algorithms.CascadeLinUCBAttack import CascadeLinUCB
from Environment import CasEnv, PbmEnv#, RealDataEnv

description = 'Run script for testing attack algorithms.'
parser = SimulationArgumentParser(description=description)

sim_args, other_args = parser.parse_all_args()

UCB_cost = pd.DataFrame()
Top_cost = pd.DataFrame()
UCB_pull = pd.DataFrame()
Top_pull = pd.DataFrame()
# print(data_cost)

# print(parser.parse_all_args())
# print(sim_args)
# print(other_args)
for key, value in other_args.items():
    if key == 'length':
        L = int(value)
    elif key == 'dimension':
        d = int(value)
    elif key == 'time':
        T =int(value)
    elif key == 'repeat':
        repeat = int(value)
    elif key == 'click_models':
        cm = value
    elif key =='synthetic':
        synthetic = value
    elif key == 'tabular':
        tabular = value
    elif key =='filename':
        filename = value
    elif key =='algorithms':
        algs = value
    # print(key,'\t',value)

starttime = time.time()
# print(T)
for envname in cm:
    print("Click model:", envname)
    for alg in algs:
        print("The attack algorithm:", alg)
        for i in range(repeat):

            seed = int(time.time() * 100) % 399
            print("Seed = %d" % seed)
            np.random.seed(seed)
            random.seed(seed)

            if envname == 'cas':
                env = CasEnv(L=L, d=d, synthetic=synthetic, tabular=tabular, filename=filename)
            elif envname == 'pbm':
                beta = [1/(k+1) for k in range(L)]
                # beta = np.ones(10) # used in MovieLens part
                env = PbmEnv(L=L, d=d, beta=beta, synthetic=synthetic, tabular=tabular, filename=filename)

            for K in [5]:
        
                if tabular and envname == 'pbm':
                    beta = [1/(k+1) for k in range(L)]
                    env = PbmEnv(L=L, d=d, beta=beta, synthetic=synthetic, tabular=tabular, filename=filename)
                
                # UCB algorithms
                if alg == 'UCB':
                    # crank = CascadeLinUCB(K, env, T)
                    # cregs, target_arm_pull = crank.run()
                    crank = CascadeLinUCB(K, env, T)
                    cregs_attack, ucb_cost, target_arm_pull_UCB = crank.attack_run()
                    UCB_cost[i] = ucb_cost
                    UCB_pull[i] = target_arm_pull_UCB

                # Top algorithm
                if alg == 'Top':
                    # trank = TopRank(K, env, T)
                    # tregs, target_arm_pull = trank.run()
                    trank = TopRank(K, env, T)
                    top_cost, target_arm_pull_TOP = trank.attack_run()
                    # top_cost, target_arm_pull_TOP = trank.attack_quit_run()
                    Top_cost[i] = top_cost
                    Top_pull[i] = target_arm_pull_TOP

    # print(data)
        if alg == 'UCB':
            UCB_cost.to_csv('plot/cost_{env}_{rep}_{time}_UCB.csv'.format(env = envname, rep = repeat, time = T))
            UCB_pull.to_csv('plot/pull_{env}_{rep}_{time}_UCB.csv'.format(env = envname, rep = repeat, time = T))
        if alg == 'Top':
            Top_cost.to_csv('plot/cost_{env}_{rep}_{time}_Top.csv'.format(env = envname, rep = repeat, time = T))
            Top_pull.to_csv('plot/pull_{env}_{rep}_{time}_Top.csv'.format(env = envname, rep = repeat, time = T))

runtime = time.time() - starttime
print(runtime)