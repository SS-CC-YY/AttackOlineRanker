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
from algorithms.CascadeUCB import CascadeUCB
from algorithms.pbmUCB import PBMUCB
from Environment import CasEnv, PbmEnv#, RealDataEnv

description = 'Run script for testing attack algorithms.'
parser = SimulationArgumentParser(description=description)

sim_args, other_args = parser.parse_all_args()


pbmucb_cost = pd.DataFrame()
casucb_cost = pd.DataFrame()
cas_Top_cost_1 = pd.DataFrame()
pbm_Top_cost_1 = pd.DataFrame()
cas_Top_cost_2 = pd.DataFrame()
pbm_Top_cost_2 = pd.DataFrame()

pbmucb_pull = pd.DataFrame()
casucb_pull = pd.DataFrame()
cas_Top_pull_1 = pd.DataFrame()
pbm_Top_pull_1 = pd.DataFrame()
cas_Top_pull_2 = pd.DataFrame()
pbm_Top_pull_2 = pd.DataFrame()

for key, value in other_args.items():
    if key == 'length':
        L = int(value)
    elif key == 'dimension':
        d = int(value)
    elif key == 'time':
        T =int(value)
    elif key == 'repeat':
        repeat = int(value)
    elif key =='synthetic':
        synthetic = value
    elif key == 'tabular':
        tabular = value
    elif key =='filename':
        filename = value
    # print(key,'\t',value)

starttime = time.time()

for alg in [1,2]:
    if alg == 1:
        print("run attack algorithm 1")
        for envname in ['cas','pbm']:
            if envname == 'cas':
                print("Cascade Click Model:")
                for i in range(repeat):
                    seed = int(time.time() * 100) % 399
                    print("Start repeat ",i+1)
                    # print("Seed = %d" % seed)
                    np.random.seed(seed)
                    random.seed(seed)
                    env = CasEnv(L=L, d=d, synthetic=synthetic, tabular=tabular, filename=filename)
                    for K in [5]:

                        # cascadeUCB
                        crank = CascadeUCB(K, env, T)
                        cregs_attack, cas_cost, target_arm_pull_cas = crank.attack_run()
                        
                        casucb_cost[i] = cas_cost
                        casucb_pull[i] = target_arm_pull_cas

                        # Toprank
                        trank = TopRank(K, env, T)
                        top_cost, target_arm_pull_top = trank.attack_run()

                        cas_Top_cost_1[i] = top_cost
                        cas_Top_pull_1[i] = target_arm_pull_top

            elif envname == 'pbm':
                print("Position Based Model:")
                for i in range(repeat):
                    seed = int(time.time() * 100) % 399
                    print("Start repeat ",i+1)
                    # print("Seed = %d" % seed)
                    np.random.seed(seed)
                    random.seed(seed)
                    beta = [1/(k+1) for k in range(L)]
                    env = PbmEnv(L=L, d=d, beta=beta, synthetic=synthetic, tabular=tabular, filename=filename)
                    for K in [5]:
                        # PBMUCB
                        prank = PBMUCB(K, env, T)
                        pregs_attack, pbm_cost, target_arm_pull_pbm = prank.attack_run()

                        pbmucb_cost[i] = pbm_cost
                        pbmucb_pull[i] = target_arm_pull_pbm

                        # Toprank
                        trank = TopRank(K, env, T)
                        top_cost, target_arm_pull_top = trank.attack_run()

                        pbm_Top_cost_1[i] = top_cost
                        pbm_Top_pull_1[i] = target_arm_pull_top
        print()
    elif alg == 2:
        print("run attack algorithm 2")
        for envname in ['cas','pbm']:
            if envname == 'cas':
                print("Cascade Click Model:")
                for i in range(repeat):
                    seed = int(time.time() * 100) % 399
                    print("Start repeat ",i+1)
                    # print("Seed = %d" % seed)
                    np.random.seed(seed)
                    random.seed(seed)
                    env = CasEnv(L=L, d=d, synthetic=synthetic, tabular=tabular, filename=filename)
                    for K in [5]:
                        # Toprank
                        trank = TopRank(K, env, T)
                        top_cost, target_arm_pull_top = trank.attack_quit_run()

                        cas_Top_cost_2[i] = top_cost
                        cas_Top_pull_2[i] = target_arm_pull_top

            elif envname == 'pbm':
                print("Position Based Model:")
                for i in range(repeat):
                    seed = int(time.time() * 100) % 399
                    print("Start repeat ",i+1)
                    # print("Seed = %d" % seed)
                    np.random.seed(seed)
                    random.seed(seed)
                    beta = [1/(k+1) for k in range(L)]
                    env = PbmEnv(L=L, d=d, beta=beta, synthetic=synthetic, tabular=tabular, filename=filename)
                    for K in [5]:
                        # Toprank
                        trank = TopRank(K, env, T)
                        top_cost, target_arm_pull_top = trank.attack_quit_run()

                        pbm_Top_cost_2[i] = top_cost
                        pbm_Top_pull_2[i] = target_arm_pull_top
        print()
runtime = time.time() - starttime
print("Finish runing, time cost:", runtime)

print("\nStart saving data:")

pbmucb_cost.to_csv('plot/cost_pbm_{rep}_{time}_pbm_alg1.csv'.format(rep = repeat, time = T))
pbmucb_pull.to_csv('plot/pull_pbm_{rep}_{time}_pbm_alg1.csv'.format(rep = repeat, time = T))
print("PBMUCB finished")

casucb_cost.to_csv('plot/cost_cas_{rep}_{time}_cas_alg1.csv'.format(rep = repeat, time = T))
casucb_pull.to_csv('plot/pull_cas_{rep}_{time}_cas_alg1.csv'.format(rep = repeat, time = T))
print("CascadeUCB finished")

cas_Top_cost_1.to_csv('plot/cost_cas_{rep}_{time}_Top_alg1.csv'.format(rep = repeat, time = T))
cas_Top_pull_1.to_csv('plot/pull_cas_{rep}_{time}_Top_alg1.csv'.format(rep = repeat, time = T))
print("Toprank with alg1 in Cascade click model finished")

pbm_Top_cost_1.to_csv('plot/cost_pbm_{rep}_{time}_Top_alg1.csv'.format(rep = repeat, time = T))
pbm_Top_pull_1.to_csv('plot/pull_pbm_{rep}_{time}_Top_alg1.csv'.format(rep = repeat, time = T))
print("Toprank with alg1 in Pbm clikc model finished")

cas_Top_cost_2.to_csv('plot/cost_cas_{rep}_{time}_Top_alg2.csv'.format(rep = repeat, time = T))
cas_Top_pull_2.to_csv('plot/pull_cas_{rep}_{time}_Top_alg2.csv'.format(rep = repeat, time = T))
print("Toprank with alg2 in Cascade click model finished")

pbm_Top_cost_2.to_csv('plot/cost_pbm_{rep}_{time}_Top_alg2.csv'.format(rep = repeat, time = T))
pbm_Top_pull_2.to_csv('plot/pull_pbm_{rep}_{time}_Top_alg2.csv'.format(rep = repeat, time = T))
print("Toprank with alg2 in Pbm clikc model finished")


