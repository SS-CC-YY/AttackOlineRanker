import numpy as np
import pickle
from sklearn.preprocessing import normalize

from ExtractFeatures import ExtractFeatures

class Environment(object):
    def __init__(self, L, d, synthetic, tabular, filename):
        super(Environment, self).__init__()
        if synthetic:
            if tabular:
                # diagonal function
                self.items = np.eye(L)
                self.means = np.random.rand(L)
                # self.means = np.arange(1,0,-1/L)
            else:
                if L == 1000:
                    with open("items.pkl", 'rb') as ifile:
                        self.items = pickle.loads(ifile.read())
                    with open("theta.pkl", 'rb') as tfile:
                        theta = pickle.loads(tfile.read())
                if L == 100:
                    with open("items_100.pkl", 'rb') as ifile:
                        self.items = pickle.loads(ifile.read())
                    with open("theta_100.pkl", 'rb') as tfile:
                        theta = pickle.loads(tfile.read())
                if L == 10:
                    with open("items_10.pkl", 'rb') as ifile:
                        self.items = pickle.loads(ifile.read())
                    with open("theta_10.pkl", 'rb') as tfile:
                        theta = pickle.loads(tfile.read())
                # self.items = self.genitems(L, d)
                # theta = self.genitems(1, d)[0]
                self.means = np.dot(self.items, theta)

        else:
            self.items, theta = ExtractFeatures(num_users=1000, num_users_in_train=100, num_items=L, d=d, filename=filename)
            self.means = np.dot(self.items, theta)

    def genitems(self, L, d):
        # Return an array of L * d, where each row is a d-dim feature vector with last entry of 1/sqrt{2}
        A = np.random.normal(0, 1, (L,d-1))
        result = np.hstack(( normalize(A, axis=1)/np.sqrt(2), np.ones((L,1))/np.sqrt(2) ))
        return result

class CasEnv(Environment):
    def __init__(self, L, d, synthetic, tabular, filename):
        super(CasEnv, self).__init__(L, d, synthetic=synthetic,tabular=tabular,filename=filename)

    def _or_func(self, v):
        return 1 - np.prod(1 - v)
    
    def feedback(self, A):
        means = self.means[A]
        x = np.random.binomial(1, means) # binomial distribution with 1 sample and probability means
        if x.sum() > 1:
            first_click = np.flatnonzero(x)[0]
            x[first_click + 1 : ] = 0
        return x, self._or_func(means)

    def get_best_reward(self, K):
        bestmeans = np.sort(self.means)[::-1][:K]
        breward = self._or_func(bestmeans)
        return breward


class PbmEnv(Environment):
    def __init__(self, L, d, beta, synthetic, tabular, filename):
        super(PbmEnv, self).__init__(L, d, synthetic=synthetic, tabular=tabular, filename=filename)
        self.beta = beta

    def feedback(self, A):
        means = self.means[A] * self.beta[:len(A)]
        return np.random.binomial(1, means), sum(means)

    def get_best_reward(self, K):
        beta = self.beta[:K]
        bestmeans = np.sort(self.means)[::-1][:K]
        breward = np.dot(beta, bestmeans)
        return breward
    

        
        