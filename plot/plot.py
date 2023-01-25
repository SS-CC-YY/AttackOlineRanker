import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_cost(pbm_UCB, 
                cas_UCB, 
                pbm_Top_1, 
                cas_Top_1,
                pbm_Top_2,
                cas_Top_2):
        fig1, ax_cas = plt.subplots()
        fig2, ax_pbm = plt.subplots()
        fig3, ax_pbm_2 = plt.subplots()
        fig4, ax_cas_2 = plt.subplots()
        xData = list(range(1,len(cas_UCB['mean'])+1))

        print(cas_Top_1)


        # ax_pbm.plot(xData, pbm_UCB['mean'],alpha=0.5,color='black',label='pbm_UCB',linewidth=1.0)
        # ax_pbm.fill_between(xData, pbm_UCB['lower'], pbm_UCB['upper'], color='black',alpha=0.4)
        ax_pbm.errorbar(xData,pbm_UCB['mean'],pbm_UCB['upper']-pbm_UCB['mean'],
                        label='pbm_UCB',errorevery=100000)

        # ax_pbm.plot(xData, pbm_Top_1['mean'],alpha=0.5,color='red',label='pbm_Top_1',linewidth=1.0)
        # ax_pbm.fill_between(xData, pbm_Top_1['lower'], pbm_Top_1['upper'], color='red',alpha=0.4)
        ax_pbm.errorbar(xData,pbm_Top_1['mean'],pbm_Top_1['upper']-pbm_Top_1['mean'],
                        label='pbm_Top_1',errorevery=120000)
        
        # ax_cas.plot(xData, cas_Top_1['mean'],alpha=0.5,color='black',label='cas_Top_1',linewidth=1.0)
        # ax_cas.fill_between(xData, cas_Top_1['lower'], cas_Top_1['upper'], color='black',alpha=0.4)
        ax_cas.errorbar(xData,cas_UCB['mean'],cas_UCB['upper']-cas_UCB['mean'],
                        label='cas_UCB',errorevery=100000)

        # ax_cas.plot(xData, cas_UCB['mean'],alpha=0.5,color='red',label='cas_UCB',linewidth=1.0)
        # ax_cas.fill_between(xData, cas_UCB['lower'], cas_UCB['upper'], color='red',alpha=0.4)
        ax_cas.errorbar(xData,cas_Top_1['mean'],cas_Top_1['upper']-cas_Top_1['mean'],
                        label='cas_Top_1',errorevery=120000)
        
        ax_pbm_2.errorbar(xData,pbm_Top_2['mean'],pbm_Top_2['upper']-pbm_Top_2['mean'],
                        label='pbm_Top',errorevery=100000)
        
        ax_cas_2.errorbar(xData,cas_Top_2['mean'],cas_Top_2['upper']-pbm_Top_2['mean'],
                        label='pbm_Top',errorevery=120000)
        

        ax_cas.legend(loc='best')
        ax_cas.set_ylabel("cost")
        ax_cas.set_xlabel("Time t")
        fig1.savefig('result_L10_T1000000_r5/total_cost_cas.pdf')

        ax_pbm.legend(loc='best')
        ax_pbm.set_ylabel("cost")
        ax_pbm.set_xlabel("Time t")
        fig2.savefig('result_L10_T1000000_r5/total_cost_pbm.png')

        ax_cas_2.legend(loc='best')
        ax_cas_2.set_ylabel("cost")
        ax_cas_2.set_xlabel("Time t")
        fig4.savefig('result_L10_T1000000_r5/total_cost_cas_alg2.png')

        ax_pbm_2.legend(loc='best')
        ax_pbm_2.set_ylabel("cost")
        ax_pbm_2.set_xlabel("Time t")
        fig3.savefig('result_L10_T1000000_r5/total_cost_pbm_alg2.png')


def plot_pull(pbm_UCB_pull,
                cas_UCB_pull,
                pbm_Top_pull_1,
                cas_Top_pull_1,
                pbm_Top_pull_2,
                cas_Top_pull_2):
        fig2, ax_pbm = plt.subplots()
        fig1, ax_cas = plt.subplots()
        fig3, ax_pbm_2 = plt.subplots()
        fig4, ax_cas_2 = plt.subplots()


        xData = list(range(1,len(cas_UCB_pull['mean'])+1))
        # ax_pbm.plot(xData, pbm_UCB_pull['mean'],alpha=0.5,color='black',label='pbm_UCB',linewidth=1.0)
        # ax_pbm.fill_between(xData, pbm_UCB_pull['lower'], pbm_UCB_pull['upper'], color='black',alpha=0.4)
        ax_pbm.errorbar(xData,pbm_UCB_pull['mean'],pbm_UCB_pull['upper']-pbm_UCB_pull['mean'],
                        label='pbm_UCB',errorevery=100000)

        # ax_pbm.plot(xData, pbm_Top_pull_1['mean'],alpha=0.5,color='red',label='pbm_Top_1',linewidth=1.0)
        # ax_pbm.fill_between(xData, pbm_Top_pull_1['lower'], pbm_Top_pull_1['upper'], color='red',alpha=0.4)
        ax_pbm.errorbar(xData,pbm_Top_pull_1['mean'],pbm_Top_pull_1['upper']-pbm_Top_pull_1['mean'],
                        label='pbm_Top',errorevery=120000)

        # ax_pbm.plot(xData, pbm_Top_pull_2['mean'],alpha=0.5,color='blue',label='pbm_Top_2',linewidth=1.0)
        # ax_pbm.fill_between(xData, pbm_Top_pull_2['lower'], pbm_Top_pull_2['upper'], color='blue',alpha=0.4)
        

        # ax_cas.plot(xData, cas_Top_pull_1['mean'],alpha=0.5,color='black',label='cas_Top_1',linewidth=1.0)
        # ax_cas.fill_between(xData, cas_Top_pull_1['lower'], cas_Top_pull_1['upper'], color='black',alpha=0.4)
        ax_cas.errorbar(xData,cas_UCB_pull['mean'],cas_UCB_pull['upper']-cas_UCB_pull['mean'],
                        label='cas_UCB',errorevery=100000)

        # ax_cas.plot(xData, cas_UCB_pull['mean'],alpha=0.5,color='red',label='cas_UCB',linewidth=1.0)
        # ax_cas.fill_between(xData, cas_UCB_pull['lower'], cas_UCB_pull['upper'], color='red',alpha=0.4)
        ax_cas.errorbar(xData,cas_Top_pull_1['mean'],cas_Top_pull_1['upper']-cas_Top_pull_1['mean'],
                        label='cas_Top_1',errorevery=120000)

        # ax_cas.plot(xData, cas_Top_pull_2['mean'],alpha=0.5,color='blue',label='cas_Top_2',linewidth=1.0)
        # ax_cas.fill_between(xData, cas_Top_pull_2['lower'], cas_Top_pull_2['upper'], color='blue',alpha=0.4)

        ax_pbm_2.errorbar(xData,pbm_Top_pull_2['mean'],pbm_Top_pull_2['upper']-pbm_Top_pull_2['mean'],
                        label='pbm_Top',errorevery=100000)
        
        ax_cas_2.errorbar(xData,cas_Top_pull_2['mean'],cas_Top_pull_2['upper']-pbm_Top_pull_2['mean'],
                        label='pbm_Top',errorevery=120000)


        ax_cas.legend(loc='best')
        ax_cas.set_ylabel("target arm pull")
        ax_cas.set_xlabel("Time t")
        fig1.savefig('result_L10_T1000000_r5/target_arm_pull_cas.png')

        ax_pbm.legend(loc='best')
        ax_pbm.set_ylabel("target arm pull")
        ax_pbm.set_xlabel("Time t")
        fig2.savefig('result_L10_T1000000_r5/target_arm_pull_pbm.png')

        ax_cas_2.legend(loc='best')
        ax_cas_2.set_ylabel("cost")
        ax_cas_2.set_xlabel("Time t")
        fig4.savefig('result_L10_T1000000_r5/target_arm_pull_cas_alg2.png')

        ax_pbm_2.legend(loc='best')
        ax_pbm_2.set_ylabel("cost")
        ax_pbm_2.set_xlabel("Time t")
        fig3.savefig('result_L10_T1000000_r5/target_arm_pull_pbm_alg2.png')

def read_data(data, data_tmp):
    # print(data_tmp)
    data['mean'] = data_tmp.mean(axis=1)
    data['upper'] = data_tmp.mean(axis=1) + data_tmp.std(axis=1)
    data['lower'] = data_tmp.mean(axis=1) - data_tmp.std(axis=1)
    return data


def main():
    # path = '/nfs/stak/users/songchen/research/AttackOnlineRanker/plot'
    path = '/nfs/stak/users/songchen/research/AttackOnlineRanker/result_L10_T1000000_r5'
    pbm_UCB = pd.DataFrame()
    cas_UCB = pd.DataFrame()
    pbm_Top_1 = pd.DataFrame()
    cas_Top_1 = pd.DataFrame()
    pbm_Top_2 = pd.DataFrame()
    cas_Top_2 = pd.DataFrame()

    pbm_UCB_pull = pd.DataFrame()
    cas_UCB_pull = pd.DataFrame()
    pbm_Top_pull_1 = pd.DataFrame()
    cas_Top_pull_1 = pd.DataFrame()
    pbm_Top_pull_2 = pd.DataFrame()
    cas_Top_pull_2 = pd.DataFrame()

    chunkSize = 10000000
    algs = []
    for root, dirs, files in os.walk(path):
        print(files)
        for i in files:
            if (i[-3:] == 'csv' and i[:4] == 'cost') and os.path.getsize(path + '/' + i) > 0:
                reader = pd.read_csv(path + '/' + i, engine='python', encoding='utf-8',index_col=0, iterator=True)
                loop = True
                chunks = []
                while loop:
                    try:
                        chunk = reader.get_chunk(chunkSize)
                        chunks.append(chunk)
                    except StopIteration:
                        loop = False
                        print("{file} load success".format(file = i))
                        # print("Iteration is stopped")
                data_tmp = pd.concat(chunks, ignore_index=True)
                data_tmp = np.cumsum(data_tmp, axis=0)
                # print(data_tmp)
                aalg = i[-8:-4]
                alg = i[-12:-9]
                click_model = i[5:8]
                if alg == 'pbm' and click_model == 'pbm':
                    pbm_UCB = read_data(pbm_UCB,data_tmp)
                if alg == 'cas' and click_model == 'cas':
                    cas_UCB = read_data(cas_UCB,data_tmp)
                if alg == 'Top' and click_model == 'cas'and aalg == 'alg1':
                    cas_Top_1 = read_data(cas_Top_1,data_tmp)
                if alg == 'Top' and click_model == 'pbm' and aalg == 'alg1':
                    pbm_Top_1 = read_data(pbm_Top_1,data_tmp)
                if alg == 'Top' and click_model == 'cas'and aalg == 'alg2':
                    cas_Top_2 = read_data(cas_Top_2,data_tmp)
                if alg == 'Top' and click_model == 'pbm' and aalg == 'alg2':
                    pbm_Top_2 = read_data(pbm_Top_2,data_tmp)
            
            
            if (i[-3:] == 'csv' and i[:4] == 'pull') and os.path.getsize(path + '/' + i) > 0:
                reader = pd.read_csv(path + '/' + i, engine='python', encoding='utf-8',index_col=0, iterator=True)
                loop = True
                chunks = []
                while loop:
                    try:
                        chunk = reader.get_chunk(chunkSize)
                        chunks.append(chunk)
                    except StopIteration:
                        loop = False
                        print("{file} load success".format(file = i))
                        # print("Iteration is stopped")
                data_tmp = pd.concat(chunks, ignore_index=True)
                data_tmp = np.cumsum(data_tmp, axis=0)
                aalg = i[-8:-4]
                alg = i[-12:-9]
                click_model = i[5:8]
                if alg == 'pbm' and click_model == 'pbm':
                    pbm_UCB_pull = read_data(pbm_UCB_pull,data_tmp)
                if alg == 'cas' and click_model == 'cas':
                    cas_UCB_pull = read_data(cas_UCB_pull,data_tmp)
                if alg == 'Top' and click_model == 'cas'and aalg == 'alg1':
                    cas_Top_pull_1 = read_data(cas_Top_pull_1,data_tmp)
                if alg == 'Top' and click_model == 'pbm' and aalg == 'alg1':
                    pbm_Top_pull_1 = read_data(pbm_Top_pull_1,data_tmp)
                if alg == 'Top' and click_model == 'cas'and aalg == 'alg2':
                    cas_Top_pull_2 = read_data(cas_Top_pull_2,data_tmp)
                if alg == 'Top' and click_model == 'pbm' and aalg == 'alg2':
                    pbm_Top_pull_2 = read_data(pbm_Top_pull_2,data_tmp)

    pbm_UCB.to_numpy
    cas_UCB.to_numpy
    pbm_Top_1.to_numpy
    cas_Top_1.to_numpy
    pbm_Top_2.to_numpy
    cas_Top_2.to_numpy

    pbm_UCB_pull.to_numpy
    cas_UCB_pull.to_numpy
    pbm_Top_pull_1.to_numpy
    cas_Top_pull_1.to_numpy
    pbm_Top_pull_2.to_numpy
    cas_Top_pull_2.to_numpy


    # print(pbm_UCB)
    # print(pbm_Top)
    plot_cost(pbm_UCB, 
    cas_UCB, 
    pbm_Top_1, 
    cas_Top_1,
    pbm_Top_2,
    cas_Top_2)
    plot_pull(pbm_UCB_pull,
    cas_UCB_pull,
    pbm_Top_pull_1,
    cas_Top_pull_1,
    pbm_Top_pull_2,
    cas_Top_pull_2)

if __name__ == "__main__":
    main()