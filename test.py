import numpy as np
from sklearn.preprocessing import normalize
import pickle
import pandas as pd
L=100
d=5
T=2000000
chunkSize = 10000000
def genitems(L, d):
    # Return an array of L * d, where each row is a d-dim feature vector with last entry of 1/sqrt{2}
    A = np.random.normal(0, 1, (L,d-1))
    result = np.hstack(( normalize(A, axis=1)/np.sqrt(2), np.ones((L,1))/np.sqrt(2) ))
    return result

items = genitems(L, d)
theta = genitems(1, d)[0]
output_item = open("items_100.pkl", 'wb')
str_item = pickle.dumps(items)
output_item.write(str_item)
output_item.close()

outpuut_theta = open("theta_100.pkl", 'wb')
str_theta = pickle.dumps(theta)
outpuut_theta.write(str_theta)
outpuut_theta.close()



# A = np.random.normal(0, 1, (L,d-1))
# result = np.hstack(( normalize(A, axis=1)/np.sqrt(2), np.ones((L,1))/np.sqrt(2) ))
# print(type(result))
# print(result)
# output_hal = open("1000_5_2000000.pkl",'wb')
# str = pickle.dumps(result)
# output_hal.write(str)
# output_hal.close()

# with open("1000_5_2000000.pkl", 'rb') as file:
#     result = pickle.loads(file.read())
# print(type(result))
# print(result)

# path = '/nfs/stak/users/songchen/research/AttackOnlineRanker'
# reader = pd.read_csv('/nfs/stak/users/songchen/research/AttackOnlineRanker/cost_cas_5_500000_Top.csv', engine='python', encoding='utf-8',index_col=0, iterator=True)
# loop = True
# chunks = []
# while loop:
#     try:
#         chunk = reader.get_chunk(chunkSize)
#         chunks.append(chunk)
#     except StopIteration:
#         loop = False
#         print("Iteration is stopped")
# data_tmp = pd.concat(chunks, ignore_index=True)
# data_tmp = np.cumsum(data_tmp, axis=0)
# print(data_tmp)