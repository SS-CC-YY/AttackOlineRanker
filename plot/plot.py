import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot(pbm_UCB, cas_UCB,pbm_Top,cas_Top,algs):
        fig, ax = plt.subplots()
        if 'UCB' in algs:
            xData = list(range(1,len(cas_UCB['mean'])+1))
            ax.plot(xData, pbm_UCB['mean'],alpha=0.5,color='red',label='pbm_UCB',linewidth=1.0)
            ax.fill_between(xData, pbm_UCB['lower'], pbm_UCB['upper'], color='red',alpha=0.4)
            ax.plot(xData, cas_UCB['mean'],alpha=0.5,color='gray',label='cas_UCB',linewidth=1.0)
            ax.fill_between(xData, cas_UCB['lower'], cas_UCB['upper'], color='gray',alpha=0.4)
        if 'Top' in algs:
            xData = list(range(1,len(cas_Top['mean'])+1))
            ax.plot(xData, pbm_Top['mean'],alpha=0.5,color='green',label='pbm_Top',linewidth=1.0)
            ax.fill_between(xData, pbm_Top['lower'], pbm_Top['upper'], color='green',alpha=0.4)
            ax.plot(xData, cas_Top['mean'],alpha=0.5,color='blue',label='cas_Top',linewidth=1.0)
            ax.fill_between(xData, cas_Top['lower'], cas_Top['upper'], color='blue',alpha=0.4)
        
        ax.legend(loc='best')
        ax.set_ylabel("cost")
        ax.set_xlabel("Time t")
        plt.savefig('total_cost.png')
def plot_pull(pbm_UCB_pull, cas_UCB_pull, pbm_Top_pull,cas_Top_pull,algs):
        fig, ax = plt.subplots()
        if 'UCB' in algs:
            xData = list(range(1,len(cas_UCB_pull['mean'])+1))
            ax.plot(xData, pbm_UCB_pull['mean'],alpha=0.5,color='red',label='pbm_UCB',linewidth=1.0)
            ax.fill_between(xData, pbm_UCB_pull['lower'], pbm_UCB_pull['upper'], color='red',alpha=0.4)
            ax.plot(xData, cas_UCB_pull['mean'],alpha=0.5,color='gray',label='cas_UCB',linewidth=1.0)
            ax.fill_between(xData, cas_UCB_pull['lower'], cas_UCB_pull['upper'], color='gray',alpha=0.4)
        if 'Top' in algs:
            xData = list(range(1,len(cas_Top_pull['mean'])+1))
            ax.plot(xData, pbm_Top_pull['mean'],alpha=0.5,color='green',label='pbm_Top',linewidth=1.0)
            ax.fill_between(xData, pbm_Top_pull['lower'], pbm_Top_pull['upper'], color='green',alpha=0.4)
            ax.plot(xData, cas_Top_pull['mean'],alpha=0.5,color='blue',label='cas_Top',linewidth=1.0)
            ax.fill_between(xData, cas_Top_pull['lower'], cas_Top_pull['upper'], color='blue',alpha=0.4)
        
        ax.legend(loc='best')
        ax.set_ylabel("cost")
        ax.set_xlabel("Time t")
        plt.savefig('target_arm_pull.png')


def main():
    path = '/nfs/stak/users/songchen/research/AttackOnlineRanker/plot'
    pbm_UCB = pd.DataFrame()
    cas_UCB = pd.DataFrame()
    pbm_Top = pd.DataFrame()
    cas_Top = pd.DataFrame()
    pbm_UCB_pull = pd.DataFrame()
    cas_UCB_pull = pd.DataFrame()
    pbm_Top_pull = pd.DataFrame()
    cas_Top_pull = pd.DataFrame()
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
                        print("Iteration is stopped")
                data_tmp = pd.concat(chunks, ignore_index=True)
                data_tmp = np.cumsum(data_tmp, axis=0)
                # for row_id in range(1,data_tmp.shape[0]):
                #     data_tmp.iloc[row_id] = data_tmp.iloc[row_id-1] + data_tmp.iloc[row_id]
                alg = i[-7:-4]
                algs.append(alg)
                click_model = i[5:8]
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
                    cas_Top['mean'] = data_tmp.mean(axis=1)
                    cas_Top['upper'] = data_tmp.mean(axis=1) + data_tmp.std(axis=1)
                    cas_Top['lower'] = data_tmp.mean(axis=1) - data_tmp.std(axis=1)
                if alg == 'Top' and click_model == 'pbm':
                    pbm_Top['mean'] = data_tmp.mean(axis=1)
                    pbm_Top['upper'] = data_tmp.mean(axis=1) + data_tmp.std(axis=1)
                    pbm_Top['lower'] = data_tmp.mean(axis=1) - data_tmp.std(axis=1)
                # print(UCB_plot)
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
                        print("Iteration is stopped")
                data_tmp = pd.concat(chunks, ignore_index=True)
                data_tmp = np.cumsum(data_tmp, axis=0)
                # for row_id in range(1,data_tmp.shape[0]):
                #     data_tmp.iloc[row_id] = data_tmp.iloc[row_id-1] + data_tmp.iloc[row_id]
                alg = i[-7:-4]
                algs.append(alg)
                click_model = i[5:8]
                if alg == 'UCB' and click_model == 'pbm':
                    pbm_UCB_pull['mean'] = data_tmp.mean(axis=1)
                    pbm_UCB_pull['upper'] = data_tmp.mean(axis=1) + data_tmp.std(axis=1)
                    pbm_UCB_pull['lower'] = data_tmp.mean(axis=1) - data_tmp.std(axis=1)
                if alg == 'UCB' and click_model == 'cas':
                    cas_UCB_pull['mean'] = data_tmp.mean(axis=1)
                    cas_UCB_pull['upper'] = data_tmp.mean(axis=1) + data_tmp.std(axis=1)
                    cas_UCB_pull['lower'] = data_tmp.mean(axis=1) - data_tmp.std(axis=1)
                if alg == 'Top' and click_model == 'cas':
                    cas_Top_pull['mean'] = data_tmp.mean(axis=1)
                    cas_Top_pull['upper'] = data_tmp.mean(axis=1) + data_tmp.std(axis=1)
                    cas_Top_pull['lower'] = data_tmp.mean(axis=1) - data_tmp.std(axis=1)
                if alg == 'Top' and click_model == 'pbm':
                    pbm_Top_pull['mean'] = data_tmp.mean(axis=1)
                    pbm_Top_pull['upper'] = data_tmp.mean(axis=1) + data_tmp.std(axis=1)
                    pbm_Top_pull['lower'] = data_tmp.mean(axis=1) - data_tmp.std(axis=1)


    pbm_UCB.to_numpy()
    cas_UCB.to_numpy()
    pbm_Top.to_numpy()
    cas_Top.to_numpy()
    pbm_UCB_pull.to_numpy()
    cas_UCB_pull.to_numpy()
    pbm_Top_pull.to_numpy()
    cas_Top_pull.to_numpy()
    # print(pbm_Top)
    plot(pbm_UCB, cas_UCB,pbm_Top,cas_Top,algs)
    plot_pull(pbm_UCB_pull, cas_UCB_pull, pbm_Top_pull,cas_Top_pull,algs)

if __name__ == "__main__":
    main()