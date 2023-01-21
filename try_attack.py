from calendar import c
from ctypes import cast
import numpy as np
import random
# import torch
from algorithms.TopRank import TopRank
from algorithms.CascadeLinUCBAttack import CascadeLinUCB
from Environment import CasEnv, PbmEnv#, RealDataEnv
from test_batch import BatchRank
import time
from matplotlib import pyplot  as plt

def main(L,d,T,envname,repeat,synthetic ,tabular,filename):

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


            # crank = CascadeLinUCB(K, env, T)
            # starttime = time.time()
            # # cregs, target_arm_pull = crank.run()
            # crank = CascadeLinUCB(K, env, T)
            # cregs_attack, cost, target_arm_pull_attack = crank.attack_run()
            # cruntime = time.time() - starttime
            # print(cruntime)
    
            # trank = TopRank(K, env, T)
            # starttime = time.time()
            # # tregs, target_arm_pull = trank.run()
            # # cost, target_arm_pull_attack = trank.attack_run()
            # cost, target_arm_pull_attack = trank.attack_quit_run()
            # # tregret = np.ones(T)*env.get_best_reward(K)-tregs_attack
            # truntime = time.time() - starttime
            # print(truntime)

            brank = BatchRank(K, env, T)
            starttime = time.time()
            a = brank.run()
            print(a)
            cost, target_arm_pull_attack = brank.run()
            bruntime = time.time() - starttime
            print(bruntime)

    # print(cost)

    # print(len(cost), np.sum(cost))
    cost_data=list(np.zeros(len(cost)))
    cost_data[0]=cost[0]
    for index in range(1, len(cost)):
        cost_data[index] = cost_data[index-1] + cost[index]
    xData = list(range(1,len(cost) + 1))

    fig, ax = plt.subplots()
    ax.plot(xData, cost_data, alpha=0.5,color='blue',label='cost',linewidth=1.0)
    ax.legend(loc='best')
    ax.set_ylabel('cost')
    ax.set_xlabel('Time t')
    plt.savefig("cost"+str(seed)+"_"+envname+"_"+str(T)+".png")

    for index in range(1, len(target_arm_pull_attack)):
        # target_arm_pull[index] = target_arm_pull[index-1] + target_arm_pull[index]
        target_arm_pull_attack[index] = target_arm_pull_attack[index-1] + target_arm_pull_attack[index]
    xData = list(range(1,len(target_arm_pull_attack) + 1))



    fig, ax2 = plt.subplots()
    ax2.plot(xData, target_arm_pull_attack,c='black',label='target arm pull')
    ax2.legend(loc='best')
    ax2.set_ylabel('target_arm_pull')
    ax2.set_xlabel('Time t')
    plt.savefig('target_arm_pull_'+str(seed)+".png")



if __name__ == "__main__":
    main(L=100,d=5,T=2000000,repeat=1,envname='cas',synthetic=True,tabular=False,filename='')
    # main(L=1000,d=5,T=500000,repeat=1,envname='cas',synthetic=True,tabular=False,filename='')
    # main(L=1000,d=5,T=200000,repeat=1,envname='cas',synthetic=False,tabular=False,filename='ml_1100_1k.npy')
    # main(L=1000,d=5,T=200000,repeat=1,envname='cas',synthetic=False,tabular=False,filename='ml_1100_1k.npy')
    # main(L=1000,d=5,T=2000000,repeat=1,envname='cas',synthetic=False,tabular=False,filename='ml_1000user_1000item.npy')
    # main(L=1000,d=5,T=200000,repeat=1,envname='pbm',synthetic=False,tabular=False,filename='ml_1000user_1000item.npy')

