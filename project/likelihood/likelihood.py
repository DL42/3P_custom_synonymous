import numpy as np
import mse_gpu
import time
from math import fsum
import scipy
from scipy import optimize


def check_alpha_pattern(SFS_size, alpha_pattern):
	previous_0 = alpha_pattern[0][0]
	previous_1 = alpha_pattern[0][1]
	if(previous_0 != 0):
		raise IndexError("alpha pattern must start at 0")
	for pattern in alpha_pattern[1:]:
		if(previous_0 >= previous_1):
			raise IndexError("alpha pattern must be strictly increasing, problem detected at {} {}".format(previous_0,previous_1))
		if(pattern[0] != previous_1): 
			raise IndexError("alpha pattern must be continuous, discontnuity detected at {} {}".format(previous_1,pattern[0]))
		previous_0 = pattern[0]
		previous_1 = pattern[1]
	
	if(previous_1 != SFS_size-2):
		raise IndexError("alpha pattern must end at {}".format((SFS_size-2)))
	
def expand_alpha(alpha_in, SFS_size, alpha_pattern):
	if(alpha_in.size != alpha_pattern.shape[0]):
		raise ValueError("alpha_in must have the same number of values as alpha_pattern has index bounds")
	alpha_out = np.array([0]*(SFS_size-2),dtype=np.float32)
	for a_in,pattern in zip(alpha_in, alpha_pattern):
		alpha_out[range(pattern[0],pattern[1])] = a_in
	return alpha_out

def theor_neutral_sfs(theta, inbreeding, alpha, mse):
	gamma = np.array([0],dtype=np.float32)
	dominance = np.array([0.5],dtype=np.float32) #effectively ignored since neutral mutations
	proportion = np.array([1],dtype=np.float32)
	return mse_gpu.calc_sfs_mse(gamma, dominance, inbreeding, proportion, theta, alpha, mse)
 
def total_likelihood(obs_test_sfs, obs_ref_sfs, theta, gamma_array, d_array, inbreeding, p_array, boundary_array, alpha, lethal_perc, mse):
	p_array=abs(p_array)
	alpha=abs(alpha)
	theta=abs(theta)
	if (gamma_array.size == 1):
		p_array=np.array([],dtype=np.float32)
    
	sum_p_array = fsum(p_array)
	loglambda = -1*np.inf
    
	if ((sum_p_array > (1 - lethal_perc)) or (lethal_perc > 1) or (lethal_perc < 0)):
		return loglambda,None,None
    
	bound_check=0
	for index, boundary in enumerate(boundary_array):
		if (boundary[0] == boundary[1]):
			gamma_array[index] = boundary[0]
		else:
			bound_check += (abs(gamma_array[index]) > abs(boundary[1])) + (abs(gamma_array[index]) < abs(boundary[0]))
    
		if (bound_check > 0):
			return loglambda,None,None
			
	theor_ref_sfs = theor_neutral_sfs(theta, inbreeding, alpha, mse)
		
	proportion = np.append(p_array,[(1 - lethal_perc) - sum_p_array])
	theor_test_sfs = mse_gpu.calc_sfs_mse(gamma_array, d_array, inbreeding, proportion, theta, alpha, mse)
	
	start_index = 0
	if (not(zero_class)):
		start_index = 1
	
	r = range(start_index,obs_ref_sfs.size)	
    
	loglambda_neu = np.dot(obs_ref_sfs[r],np.log(theor_ref_sfs[r]))
	loglambda_sel = np.dot(obs_test_sfs[r],np.log(theor_test_sfs[r]))

	loglambda = loglambda_neu + loglambda_sel
   
	return loglambda,theor_test_sfs,theor_ref_sfs

def re_param_neutral_alpha(x):
	theta_site = x[ap_size]
	p_array = np.array([],dtype=np.float32)
	alpha = my_expand_alpha(x[0:ap_size])
	g_array = np.array([0],dtype=np.float32)
	d_array = np.array([0.5],dtype=np.float32)
	b_array = np.array([[0,0]])
	
	result = total_likelihood(obs_ref_sfs, obs_ref_sfs, theta_site, g_array, d_array, inbreeding, p_array, b_array, alpha, 0, mse)
	return -1*result[0]
    
def re_param(x,fixed_alpha):
	theta_site = x[ap_size]
	p_array = x[(ap_size+1):(ap_size+1+p_size)]
	if (free_alpha and not(neutral_alpha)):
		alpha = my_expand_alpha(x[0:ap_size])
	else:
		alpha = my_expand_alpha(fixed_alpha[0:ap_size])
	g_array = x[(ap_size+1+p_size):x.size]
	d_array = np.array([0.5]*g_array.size,dtype=np.float32)
	result = total_likelihood(obs_test_sfs, obs_ref_sfs, theta_site, g_array, d_array, inbreeding, p_array, boundary_array, alpha, 0, mse) 
	print(g_array,p_array,result[0])
	return -1*result[0]
    
def re_param_neu(x):
	theta_site = x[ap_size]
	p_array = np.array([],dtype=np.float32)
	if (free_alpha and not(neutral_alpha)):
		alpha = my_expand_alpha(x[0:ap_size])
	else:
		alpha = my_expand_alpha(fixed_alpha2[0:ap_size])
	g_array = np.array([0],dtype=np.float32)
	d_array = np.array([0.5],dtype=np.float32)
	b_array = np.array([[0,0]])
	result = total_likelihood(obs_test_sfs, obs_ref_sfs, theta_site, g_array, d_array, inbreeding, p_array, b_array, alpha, 0, mse)
	loglambda = -1*result[0]
	return -1*result[0]
    
def re_param_neu_lethal(x):
	theta_site = x[ap_size]
	p_array = 1.0 - x[(x.size-1)]
	if (free_alpha and not(neutral_alpha)):
		alpha = my_expand_alpha(x[0:ap_size])
	else:
		alpha = my_expand_alpha(fixed_alpha2[0:ap_size])
	g_array = np.array([0],dtype=np.float32)
	d_array = np.array([0.5],dtype=np.float32)
	b_array = np.array([[0,0]])
	lethal_perc = x[x.size-1]
	result = total_likelihood(obs_test_sfs, obs_ref_sfs, theta_site, g_array, d_array, inbreeding, p_array, b_array, alpha, lethal_perc, mse)
	return -1*result[0]
    
def re_param_lethal(x):
	theta_site = x[ap_size]
	p_array=x[(ap_size+1):(ap_size+1+p_size)]
	if (free_alpha and not(neutral_alpha)):
		alpha = my_expand_alpha(x[0:ap_size])
	else:
		alpha = my_expand_alpha(fixed_alpha2[0:ap_size])
	g_array = x[(ap_size+1+p_size):(x.size-1)]
	d_array = np.array([0.5]*g_array.size,dtype=np.float32)
	lethal_perc = x[x.size-1]
	result = total_likelihood(obs_test_sfs, obs_ref_sfs, theta_site, g_array, d_array, inbreeding, p_array, boundary_array, alpha, lethal_perc, mse)
	return -1*result[0]

def maximum_likelihood(obs_test_sfs, obs_ref_sfs, file_name, initial_gamma_array, boundary_array, initial_p_array, initial_theta_site, initial_lethal_array, alpha_pattern, free_alpha, neutral_alpha, fold, zero_class, N_chrome, optimizer = 'Nelder-Mead'):

	SFS_size = obs_test_sfs.size
	check_alpha_pattern(SFS_size, alpha_pattern)
	my_expand_alpha = lambda x: expand_alpha(x, SFS_size, alpha_pattern)
	ap_size = alpha_pattern.shape[0]
	init_alpha = np.ones(ap_size)
	p_size = initial_p_array.size
	g_size = initial_gamma_array.size
	num_test_sites = fsum(obs_test_sfs)
	num_ref_sites = fsum(obs_ref_sfs)
	n_samp = obs_test_sfs.size
	if(fold): 
		n_samp = 2*(n_samp-1)
	inbreeding = 0.0
	dominance = np.array([0.5]*initial_gamma_array.size,dtype=np.float32)
	#mse = mse_gpu.MSE(n_samp,N_chrome,fold,zero_class)

	x_neu_alpha = np.append(init_alpha,initial_theta_site)
	maxfev = 100000*(x_neu_alpha.size+p_size+g_size)

	if(neutral_alpha and free_alpha):
		x_max_neu_alpha = scipy.optimize.minimize(re_param_neutral_alpha, x_neu_alpha, method=optimizer, options={'maxfev': maxfev, 'disp': True})
		init_alpha = x_max_neu_alpha["x"][:ap_size]
		print(x_max_neu_alpha)
	x = np.concatenate((np.ones(ap_size),np.array([initial_theta_site]),initial_p_array,initial_gamma_array)) 

	x_max = scipy.optimize.minimize(re_param, x, (init_alpha,), method=optimizer, options={'maxfev': maxfev, 'disp': True, 'xatol': 0.0000001, 'fatol': 0.1})
	if(free_alpha and not(neutral_alpha)):
		max_alpha = my_expand_alpha(x_max["x"][:ap_size])
	else:
		max_alpha = my_expand_alpha(init_alpha) 

	max_theta_site = abs(x_max["x"][ap_size])   
	max_p_array = abs(x_max["x"][(ap_size+1):(ap_size+1+p_size)]) 
	max_gamma = x_max["x"][(ap_size+1+p_size):x.size] 
	#print(x_max) 
	max_likelihood,theor_test_sfs,theor_ref_sfs = total_likelihood(obs_test_sfs, obs_ref_sfs, max_theta_site, max_gamma, dominance, inbreeding, max_p_array, boundary_array, max_alpha, 0, mse) 
    
	x_neu = np.append(np.ones(ap_size),initial_theta_site)
	x_max_neu = scipy.optimize.minimize(re_param_neu, x_neu, method=optimizer, options={'maxfev': maxfev}) 
	all_neutral_lik = -1*x_max_neu["fun"]
	max_theta_site_neu = x_max_neu["x"][ap_size]
    
	if(zero_class):
		x_neu_lethal = np.concatenate((np.ones(ap_size),np.array([initial_theta_site]),initial_lethal_array[0:1]))
		x_max_neu_lethal = scipy.optimize.minimize(re_param_neu_lethal, x_neu_lethal, method=optimizer, options={'maxfev': maxfev})   
		max_neutral_lethal_lik = -1*x_max_neu_lethal["fun"]
		max_neu_lethal_perc = x_max_neu_lethal["x"][x_max_neu_lethal["x"].size-1]
		max_theta_site_neu_lethal = x_max_neu_lethal["x"][ap_size]
        
		x_lethal = np.concatenate((np.ones(ap_size),np.array([initial_theta_site]),initial_p_array,initial_gamma_array,initial_lethal_array[1:2]))  
		x_max_lethal = scipy.optimize.minimize(re_param_lethal, x_lethal, method=optimizer, options={'maxfev': maxfev})
		max_lethal_perc = x_max_lethal["x"][x_max_lethal["x"].size-1]
		max_lethal_lik = -1*x_max_lethal["fun"]
		max_theta_site_lethal = x_max_lethal["x"][ap_size]
		max_lethal_p_array = x_max_lethal["x"][(ap_size+1):(ap_size+1+p_size)]
		max_lethal_gamma = x_max_lethal["x"][(ap_size+1+p_size):(x_max_lethal["x"].size-1)]
    
	f = open(file_name,"a")
	f.write('{}_selection_classes\t'.format(initial_gamma_array.size))
	if (zero_class):
		f.write('zero_class\t')
	else:
		f.write('shape_only\t')
    
	if (free_alpha and not(neutral_alpha)):
		f.write('freeAlpha6Bins\t') #add alpha_pattern to description
	else:
		if (neutral_alpha and free_alpha):
			f.write('neutAlpha6Bins\t')
		else:
			f.write('fixedAlpha_one\t')
    
	f.write('({}'.format(initial_gamma_array[0]))
	for gamma in initial_gamma_array[1:]:
		f.write(',{}'.format(gamma))
    
	f.write(';{}'.format(boundary_array[0]))
	for boundary in boundary_array[1:]:
		f.write(',{}'.format(boundary))
    
	f.write(';{}'.format(initial_p_array[0]))
	for prop in initial_p_array[1:]:
		f.write(',{}'.format(prop))   
	f.write(',{};'.format((1 - fsum(initial_p_array))))
    
	f.write('{:6.4f};'.format(initial_theta_site))
	f.write('{},{},{})\t('.format(initial_lethal_array[0],initial_lethal_array[1],N_chrome))
    
	stop = max_gamma.size - 1
	pfinal = 1
	if(stop != 0):
		pfinal = 1 - fsum(max_p_array)
		for index, (boundary, p) in enumerate(zip(boundary_array[:stop], max_p_array)):
			if (boundary[0] == boundary[1]):
				max_gamma[index] = boundary[0]
				print(max_gamma)
			f.write('{} {:6.4f}, '.format(max_gamma[index],p))
    	
	if (boundary_array[stop,0] == boundary_array[stop,1]):
		max_gamma[stop] = boundary_array[stop,0]
    
	if (zero_class):
		f.write('{} {:6.4f}, theta {:6.4f}): '.format(max_gamma[stop],pfinal,max_theta_site))
	else:
		f.write('{} {:6.4f}): '.format(max_gamma[stop],pfinal))
    
	f.write('{:6.4f}\t'.format(max_likelihood))
	if (zero_class):
		f.write('(neutral 1.0, theta {:6.4f}): {:6.4f}\t'.format(max_theta_site_neu,all_neutral_lik))
	else:
		f.write('(neutral 1.0): {:6.4f}\t'.format(all_neutral_lik))
    
	if (zero_class):
		f.write('(neutral {:6.4f}, lethal {:6.4f}, theta {:6.4f}): {:6.4f}\t('.format((1 - max_neu_lethal_perc),max_neu_lethal_perc,max_theta_site_neu_lethal,max_neutral_lethal_lik))
		pfinal = 1 - max_lethal_perc
		if(stop != 0):
			pfinal = 1 - fsum(max_lethal_p_array) - max_lethal_perc	
			for index, (boundary, lethal_p) in enumerate(zip(boundary_array[:stop], max_lethal_p_array)):
				if (boundary[0] == boundary[1]):
					max_lethal_gamma[index] = boundary[0]
				f.write('{} {:6.4f}, '.format(max_lethal_gamma[index], lethal_p))
		if (boundary_array[stop,0] == boundary_array[stop,1]):
			max_lethal_gamma[stop]=boundary_array[stop,0]
							
		f.write('{} {:6.4f}, lethal {:6.4f}, theta {:6.4f}): {:6.4f}'.format(max_lethal_gamma[stop],pfinal,max_lethal_perc,max_theta_site_lethal,max_lethal_lik))
    
	f.write('\n')
	f.close()
	return max_likelihood,theor_test_sfs*num_test_sites,theor_ref_sfs*num_ref_sites,max_alpha,max_gamma,max_theta_site,max_p_array

if __name__ == '__main__':
	alpha_pattern = np.array([[0,1],[1,3],[3,7],[7,15],[15,31],[31,79]])
	ap_size = alpha_pattern.shape[0]
	fold = True
	zero_class = True
	free_alpha = True
	neutral_alpha = True
	N_chrome = 40000
	gamma_strength = -200
	mse2 = mse_gpu.MSE(160,N_chrome,fold,True)
	gamma = np.array([0,gamma_strength],dtype=np.float32)
	dominance = np.array([0.5,0.5],dtype=np.float32) #effectively ignored since inbreeding is set to 1
	proportion = np.array([0.85,0.15],dtype=np.float32)
	p_size = proportion.size -1
	fixed_alpha2 = np.array([1.0]*6)

	alpha = np.array([1]*79,dtype=np.float32)
	theta = 0.01
	inbreeding = 0.0
	num_sites = 2*pow(10,7)
	selected = mse_gpu.calc_sfs_mse(gamma,dominance,inbreeding,proportion,theta,alpha,mse2)
	obs_test_sfs = num_sites*selected
	neutral = theor_neutral_sfs(theta, inbreeding, alpha, mse2)
	obs_ref_sfs = num_sites*neutral
	
	mse = mse_gpu.MSE(160,N_chrome,fold,zero_class)
	
	file_name = "test_ml.txt"
	boundary_array = np.array([[0,0],[0,-250]])
	initial_gamma_array = np.array([0,-100],dtype=np.float32)
	initial_p_array = np.array([0.65],dtype=np.float32)
	initial_theta_site = 0.005
	initial_lethal_array = np.array([0.01,0.01],dtype=np.float32)
	SFS_size = obs_test_sfs.size
	my_expand_alpha = lambda x: expand_alpha(x, SFS_size, alpha_pattern)
	before = time.time()
	ml_results = maximum_likelihood(obs_test_sfs, obs_ref_sfs, file_name, initial_gamma_array, boundary_array, initial_p_array, initial_theta_site, initial_lethal_array, alpha_pattern, free_alpha, neutral_alpha, fold, zero_class, N_chrome, optimizer = 'Nelder-Mead')
	process_time = time.time() - before
	print(process_time,ml_results)
	print(obs_test_sfs,obs_ref_sfs)
	p_array = proportion[0:1]
	results = total_likelihood(obs_test_sfs, obs_ref_sfs, theta, gamma, dominance, inbreeding, p_array, boundary_array, alpha, 0, mse)
	print(results[0],results[1]*num_sites,results[2]*num_sites)
    
