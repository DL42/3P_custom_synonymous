import numpy as np
import mse_gpu
import time
mse = mse_gpu.MSE(160,40000,1)
gamma = np.array([0],dtype=np.float32)
sel_prop = np.array([1],dtype=np.float32)
theta = 0.01
num_sites = 2*pow(10,7)
fold = True
zero_class = True
before = time.time()
result = mse_gpu.calc_sfs_mse(gamma,sel_prop,theta,num_sites,fold,zero_class,mse)
process_time = time.time() - before
print(process_time)