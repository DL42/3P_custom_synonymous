import numpy as np
import mse_gpu
import time
from math import fsum
import scipy
from scipy import optimize

if __name__ == '__main__':
	fold = True
	N_chrome = 40000
	gamma_strength = -200
	mse = mse_gpu.MSE(160,N_chrome,fold,True)
	gamma = np.array([0,gamma_strength],dtype=np.float64)
	dominance = np.array([0.5,0.5],dtype=np.float64) #effectively ignored since inbreeding is set to 1
	proportion = np.array([0.85,0.15],dtype=np.float64)
	
	alpha = np.array([1]*79,dtype=np.float64)
	theta = 0.01
	inbreeding = 0.0
	num_sites = 2*pow(10,7)
	selected = mse_gpu.calc_sfs_mse(gamma,dominance,inbreeding,proportion,theta,alpha,mse)
	my_test_sfs = num_sites*selected
	
	file_name = "sfs_test_out.txt"
	f = open(file_name,'w')
	for freq in my_test_sfs:
		f.write("{}\n".format(freq))
	f.close()
	
	gamma = np.array([0],dtype=np.float64)
	dominance = np.array([0.5],dtype=np.float64) #effectively ignored since neutral mutations
	proportion = np.array([1],dtype=np.float64)
	neutral = mse_gpu.calc_sfs_mse(gamma, dominance, inbreeding, proportion, theta, alpha, mse)
	my_ref_sfs = num_sites*neutral
	
	file_name = "sfs_ref_out.txt"
	f = open(file_name,'w')
	for freq in my_ref_sfs:
		f.write("{}\n".format(freq))
	f.close()