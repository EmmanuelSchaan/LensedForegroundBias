import numpy as np
import matplotlib.pyplot as plt

path = "./advact/cmblensrecnoise_lmax3000.txt"

data = np.genfromtxt(path)
L = data[:,0]
# diagonal covariances
N_TT = data[:,1]
N_TE = data[:,2]
N_TB = data[:,3]
N_EE = data[:,4]
N_EB = data[:,5]
# non-diagonal covariances
N_TT_TE = data[:,6]
N_TT_TB = data[:,7]
N_TT_EE = data[:,8]
N_TT_EB = data[:,9]
N_TE_TB = data[:,10]
N_TE_EE = data[:,11]
N_TE_EB = data[:,12]
N_TB_EE = data[:,13]
N_TB_EB = data[:,14]
N_EE_EB = data[:,15]
# variance of mv estimator
N_mv = data[:,16]


# convert from dd to kappakappa
N_kappakappa = L*(L+1.)*N_mv/4.

plt.loglog(L, N_kappakappa)
plt.show()
