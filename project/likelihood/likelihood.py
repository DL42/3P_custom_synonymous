import numpy as np
import mse_gpu
import time
from math import fsum

fold = True
zero_class = True
mse = mse_gpu.MSE(160,40000,fold,zero_class)
gamma = np.array([0,-200],dtype=np.float32)
dominance = np.array([0.5,0.5],dtype=np.float32) #effectively ignored since inbreeding is set to 1
proportion = np.array([0.85,0.15],dtype=np.float32)
alpha = np.array([1]*79,dtype=np.float32)
theta = 0.01
inbreeding = 0.0
num_sites = 2*pow(10,7)
before = time.time()
selected = mse_gpu.calc_sfs_mse(gamma,dominance,inbreeding,proportion,theta,alpha,mse)
process_time = time.time() - before
print(process_time)
print(selected*num_sites)
print(sum(selected))

def theor_neutral(theta, alpha, mse):
	gamma = np.array([0],dtype=np.float32)
	dominance = np.array([0.5],dtype=np.float32) #effectively ignored since inbreeding is set to 1
	proportion = np.array([1],dtype=np.float32)
	inbreeding = 0
	return mse_gpu.calc_sfs_mse(gamma,dominance,inbreeding,proportion,theta,alpha,mse)

before = time.time()
neutral = theor_neutral(theta, alpha, mse)
process_time = time.time() - before
print(process_time)
print(neutral*num_sites)
print(sum(neutral))

before = time.time()
alpha = np.array([0.9]*79,dtype=np.float32)
renorm = mse_gpu.renormalize_SFS(neutral, alpha, mse)
process_time = time.time() - before
print(process_time)
print(renorm*num_sites)
print(sum(renorm))

# num_test_sites=sum(obs_test_sfs)
# num_ref_sites=sum(obs_ref_sfs)
# n_samp = length(obs_test_sfs) - 1 
# dominance = np.array([0.5]*proportion.size,dtype=np.float32)
# inbreeding = 0.0
 
def total_likelihood(obs_test_sfs, obs_ref_sfs, theta, gamma_array, d_array, inbreeding, p_array, boundary_array, alpha, lethal_perc, mse):
    p_array=abs(p_array)
    alpha=abs(alpha)
    theta=abs(theta)
    if (gamma_array.size == 1):
        p_array=np.array([],dtype=np.float32)
    
    sum_p_array = fsum(p_array)
    loglambda = float("-inf")
    
    if ((sum_p_array > (1 - lethal_perc)) or (lethal_perc > 1) or (lethal_perc < 0)):
        return loglambda,alpha,None,None
    
    bound_check=0
    for gamma, boundary in zip(gamma_array, boundary_array):
        if (boundary[0] == boundary[1]):
            gamma = boundary[0]
        else:
            bound_check += (abs(gamma) > abs(boundary[1])) + (abs(gamma) < abs(boundary[0]))
    
    if (bound_check > 0):
        return loglambda,alpha,None,None

    theor_ref_sfs = theor_neutral(theta, alpha, mse)
    
    proportion = np.append(p_array,[(1 - lethal_perc) - sum_p_array])
    theor_test_sfs = mse_gpu.calc_sfs_mse(gamma_array,d_array,inbreeding,proportion,theta,alpha,mse)
    
    start_index = 0
    if (not(zero_class)):
    	start_index = 1
    
    r = range(start_index,obs_ref_sfs.size)	
    
    loglambda_neu = np.dot(obs_ref_sfs[r],np.log(theor_ref_sfs[r]))
    loglambda_sel = np.dot(obs_test_sfs[r],np.log(theor_test_sfs[r]))

    loglambda = loglambda_neu + loglambda_sel
   
    return loglambda,alpha,theor_test_sfs,theor_ref_sfs

neu = neutral*num_sites
sel = selected*num_sites
before = time.time()
start_index = 0
if (not(zero_class)):
    start_index = 1
    
r = range(start_index,neu.size)
loglambda_neu = np.dot(neu[r],np.log(neutral[r]))
loglambda_sel = np.dot(sel[r],np.log(selected[r]))
loglambda = loglambda_neu + loglambda_sel
process_time = time.time() - before
print("WTF",process_time,loglambda)

alpha = np.array([1]*79,dtype=np.float32)
boundary_array = np.array([[0.0,0.0],[0.0,-500]])
p_array = np.array([0.85])
sel = selected*num_sites
neu = neutral*num_sites
before = time.time()
lik = total_likelihood(sel, neu, 0.01, gamma, dominance, inbreeding, p_array, boundary_array, alpha, 0, mse)
process_time = time.time() - before
print(process_time)
print(lik[0])