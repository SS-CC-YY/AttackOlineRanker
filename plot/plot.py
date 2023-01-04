import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot(pbm_UCB, cas_UCB,pbm_Top,cas_Top,pbm_TS,cas_TS,algs):
        fig, ax = plt.subplots()
        if 'UCB' in algs:
            xData = list(range(1,len(pbm_UCB['mean'])+1))
            ax.plot(xData, pbm_UCB['mean'],alpha=0.5,color='red',label='pbm_UCB',linewidth=1.0)
            ax.fill_between(xData, pbm_UCB['lower'], pbm_UCB['upper'], color='red',alpha=0.4)
            ax.plot(xData, cas_UCB['mean'],alpha=0.5,color='gray',label='cas_UCB',linewidth=1.0)
            ax.fill_between(xData, cas_UCB['lower'], cas_UCB['upper'], color='gray',alpha=0.4)
        if '_TS' in algs:
            xData = list(range(1,len(pbm_TS['mean'])+1))
            ax.plot(xData, pbm_TS['mean'],alpha=0.5,color='blue',label='pbm_TS',linewidth=1.0)
            ax.fill_between(xData, pbm_TS['lower'], pbm_TS['upper'], color='blue',alpha=0.4)
            ax.plot(xData, cas_TS['mean'],alpha=0.5,color='yellow',label='cas_TS',linewidth=1.0)
            ax.fill_between(xData, cas_TS['lower'], cas_TS['upper'], color='yellow',alpha=0.4)
        if 'Top' in algs:
            xData = list(range(1,len(pbm_Top['mean'])+1))
            ax.plot(xData, pbm_Top['mean'],alpha=0.5,color='green',label='pbm_Top',linewidth=1.0)
            ax.fill_between(xData, pbm_Top['lower'], pbm_Top['upper'], color='green',alpha=0.4)
                
        
        ax.legend(loc='best')
        ax.set_ylabel("cost")
        ax.set_xlabel("Time t")
        plt.savefig('total_cost_test.png')


def main():
    path = '/nfs/stak/users/songchen/research/AttackOnlineRanker/plot'
    pbm_UCB = pd.DataFrame()
    cas_UCB = pd.DataFrame()
    pbm_Top = pd.DataFrame()
    cas_Top = pd.DataFrame()
    pbm_TS = pd.DataFrame()
    cas_TS = pd.DataFrame()
    algs = []
    for root, dirs, files in os.walk(path):
        # print(files)
        for i in files:
            if (i[-3:] == 'csv' and i[:4] == 'cost') and os.path.getsize(path + '/' + i) > 0:
                data_tmp = pd.read_csv(path + '/' + i, engine='python', encoding='utf-8',index_col=0)
                for row_id in range(1,data_tmp.shape[0]):
                    data_tmp.iloc[row_id] = data_tmp.iloc[row_id-1] + data_tmp.iloc[row_id]
                alg = i[-7:-4]
                algs.append(alg)
                click_model = i[5:8]
                # print(click_model)
                content = i[:4]
                if alg == 'UCB' and click_model == 'pbm':
                    pbm_UCB['mean'] = data_tmp.mean(axis=1)
                    pbm_UCB['upper'] = data_tmp.mean(axis=1) + data_tmp.std(axis=1)
                    pbm_UCB['lower'] = data_tmp.mean(axis=1) - data_tmp.std(axis=1)
                if alg == 'UCB' and click_model == 'cas':
                    cas_UCB['mean'] = data_tmp.mean(axis=1)
                    cas_UCB['upper'] = data_tmp.mean(axis=1) + data_tmp.std(axis=1)
                    cas_UCB['lower'] = data_tmp.mean(axis=1) - data_tmp.std(axis=1)
                if alg == 'Top' and click_model == 'cas':
                    cas_Top['mean'] = data_tmp.mean(axis=1)*10
                    cas_Top['upper'] = data_tmp.mean(axis=1)*10 + data_tmp.std(axis=1)*10
                    cas_Top['lower'] = data_tmp.mean(axis=1)*10 - data_tmp.std(axis=1)*10
                if alg == 'Top' and click_model == 'pbm':
                    pbm_Top['mean'] = data_tmp.mean(axis=1)*10
                    pbm_Top['upper'] = data_tmp.mean(axis=1)*10 + data_tmp.std(axis=1)*10
                    pbm_Top['lower'] = data_tmp.mean(axis=1)*10 - data_tmp.std(axis=1)*10
                if alg == '_TS' and click_model == 'pbm':
                    pbm_TS['mean'] = data_tmp.mean(axis=1)
                    pbm_TS['upper'] = data_tmp.mean(axis=1) + data_tmp.std(axis=1)
                    pbm_TS['lower'] = data_tmp.mean(axis=1) - data_tmp.std(axis=1)
                if alg == '_TS' and click_model == 'cas':
                    cas_TS['mean'] = data_tmp.mean(axis=1)
                    cas_TS['upper'] = data_tmp.mean(axis=1) + data_tmp.std(axis=1)
                    cas_TS['lower'] = data_tmp.mean(axis=1) - data_tmp.std(axis=1)
                # print(UCB_plot)

    pbm_UCB.to_numpy()
    cas_UCB.to_numpy()
    pbm_TS.to_numpy()
    cas_TS.to_numpy()
    pbm_Top.to_numpy()
    cas_Top.to_numpy()
    # print(pbm_Top)
    # print(algs)
    plot(pbm_UCB, cas_UCB,pbm_Top,cas_Top,pbm_TS,cas_TS,algs)


if __name__ == "__main__":
    main()