import numpy as np
import mse_gpu
import time
fold = False
zero_class = True
mse = mse_gpu.MSE(160,40000,fold,zero_class)
gamma = np.array([0,-200],dtype=np.float32)
dominance = np.array([0,0],dtype=np.float32) #effectively ignored since inbreeding is set to 1
proportion = np.array([0.85,0.15],dtype=np.float32)
alpha = np.array([1]*158,dtype=np.float32)
theta = 0.01
inbreeding = 1.0
num_sites = 2*pow(10,7)
before = time.time()
result = mse_gpu.calc_sfs_mse(gamma,dominance,inbreeding,proportion,theta,alpha,mse)
process_time = time.time() - before
print(process_time)
print(result*num_sites)
print(sum(result))

def theor_neutral(mse, theta):
	gamma = np.array([0],dtype=np.float32)
	dominance = np.array([0],dtype=np.float32) #effectively ignored since inbreeding is set to 1
	proportion = np.array([1],dtype=np.float32)
	alpha = np.array([1]*158,dtype=np.float32)
	inbreeding = 1
	return mse_gpu.calc_sfs_mse(gamma,dominance,inbreeding,proportion,theta,alpha,mse)

before = time.time()
result = theor_neutral(mse, theta)
process_time = time.time() - before
print(process_time)
print(result*num_sites)
print(sum(result))

before = time.time()
alpha = np.array([0.9]*158,dtype=np.float32)
result = mse_gpu.renormalize_SFS(result, alpha, mse)
process_time = time.time() - before
print(process_time)
print(result*num_sites)
print(sum(result))
# num_test_sites=sum(obs_test_sfs)
# num_ref_sites=sum(obs_ref_sfs)
# n_samp = length(obs_test_sfs) - 1
# if(fold): n_samp = 2*(length(obs_test_sfs) - 1)
# 

# 
# def total_likelihood(obs_test_sfs, obs_ref_sfs, n_samp, unnorm_theor_ref_sfs, mse, theta, gamma_array, p_array, boundary_array, alpha, lethal_perc):
#     p_array=abs(p_array)
#     alpha=abs(alpha)
#     theta=abs(theta)
#     if (length(gamma_array) == 1):
#         p_array=np.array([],dtype=np.float32)
#     
#     if ((sum(p_array) > (1 - lethal_perc)) or (lethal_perc > 1) or (lethal_perc < 0)):
#         loglambda=- 1 / 0
#         return loglambda,alpha,None,None
#     
#     bool=0
#     for index1 in arange(1,length(gamma)).reshape(-1):
#         if (boundary_array(index1,2) == boundary_array(index1,1)):
#             gamma_array[index1]=boundary_array(index1,2)
#         else:
#             bool=bool + (gamma_array(index1) < boundary_array(index1,2)) + (gamma_array(index1) > boundary_array(index1,1))
#     
#     if (bool > 0):
#         # bool,gamma_array
#         loglambda=- 1 / 0
#         return loglambda,alpha,None,None
#     
#     mu_persite=theta / (dot(4,N_pop))
#     g_mix_neu=theor_SFS(concat([0,1.0]),mu_persite,num_ref_sites,N_pop,n_samp,P_samp)
#     
#     theor_ref_sfs=zeros((length(obs_ref_sfs) - 1),1)
#     theor_ref_sfs[1]=dot(1,(g_mix_neu(1) + g_mix_neu(n_samp - 1)))
#     for k in arange(2,(length(obs_ref_sfs) - 1)).reshape(-1):
#         theor_ref_sfs[k]=dot(alpha((k - 1)),(g_mix_neu(k) + g_mix_neu(n_samp - k)))
#         if (k == n_samp - k):
#             theor_ref_sfs[k]=dot(alpha((k - 1)),g_mix_neu(k))
#     
#     
#     if (zero_class):
#         #can't use m_samp because alphas, so re-sum number of neutral SNPs
#         g0_mix_neu=num_ref_sites - sum(theor_ref_sfs)
#         theor_ref_sfs=concat([[g0_mix_neu],[theor_ref_sfs]])
#         rho_mix_neu_fold=theor_ref_sfs / num_ref_sites
#     else:
#         rho_mix_neu=theor_ref_sfs / sum(theor_ref_sfs)
#         rho_mix_neu_fold=concat([[0],[rho_mix_neu]])
#     
#     
#     gamma_array_comb=concat([[gamma_array],[concat([p_array,(1 - lethal_perc) - sum(p_array)])]]).T
#     g_mix_sel=theor_SFS(gamma_array_comb,mu_persite,num_test_sites,N_pop,n_samp,P_samp)
# 
#     theor_test_sfs=zeros((length(obs_ref_sfs) - 1),1)
#     theor_test_sfs[1]=dot(1,(g_mix_sel(1) + g_mix_sel(n_samp - 1)))
#     for k in arange(2,(length(obs_ref_sfs) - 1)).reshape(-1):
#         theor_test_sfs[k]=dot(alpha((k - 1)),(g_mix_sel(k) + g_mix_sel(n_samp - k)))
#         if (k == n_samp - k):
#             theor_test_sfs[k]=dot(alpha((k - 1)),g_mix_sel(k))
#     
#     
#     if (zero_class):
#         #can't use m_samp because alphas, so re-sum number of selected SNPs
#         g0_mix_sel=num_test_sites - sum(theor_test_sfs)
#         theor_test_sfs=concat([[g0_mix_sel],[theor_test_sfs]])
#         rho_mix_sel_fold=theor_test_sfs / num_test_sites
#     else:
#         rho_mix_sel=theor_test_sfs / sum(theor_test_sfs)
#         rho_mix_sel_fold=concat([[0],[rho_mix_sel]])
#     
#     
#     loglambda_neu=0
#     loglambda_sel=0
#     if (zero_class):
#         loglambda_neu=dot(obs_ref_sfs(1),log(rho_mix_neu_fold(1)))
#         loglambda_sel=dot(obs_test_sfs(1),log(rho_mix_sel_fold(1)))
#     
#     
#     
#     for k in arange(2,length(obs_ref_sfs)).reshape(-1):
#         loglambda_neu=loglambda_neu + dot(obs_ref_sfs(k),log(rho_mix_neu_fold(k)))
#     
#     
#     for k in arange(2,length(obs_test_sfs)).reshape(-1):
#         loglambda_sel=loglambda_sel + dot(obs_test_sfs(k),log(rho_mix_sel_fold(k)))
#     
#     
#     loglambda=loglambda_neu + loglambda_sel
#     return loglambda,alpha,theor_test_sfs,theor_ref_sfs