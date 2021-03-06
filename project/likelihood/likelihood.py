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
	alpha_out = np.array([0]*(SFS_size-2),dtype=np.float64)
	for a_in,pattern in zip(alpha_in, alpha_pattern):
		alpha_out[range(pattern[0],pattern[1])] = a_in
	return alpha_out

def theor_neutral_sfs(theta, inbreeding, alpha, mse):
	gamma = np.array([0],dtype=np.float64)
	dominance = np.array([0.5],dtype=np.float64) #effectively ignored since neutral mutations
	proportion = np.array([1],dtype=np.float64)
	return mse_gpu.calc_sfs_mse(gamma, dominance, inbreeding, proportion, theta, alpha, mse)
 
def total_likelihood(obs_test_sfs, obs_ref_sfs, theta, gamma_array, d_array, inbreeding, p_array, boundary_array, alpha, lethal_perc, mse):
	p_array=abs(p_array)
	alpha=abs(alpha)
	theta=abs(theta)
	if (gamma_array.size == 1):
		p_array=np.array([],dtype=np.float64)
    
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
	#print(obs_ref_sfs,obs_test_sfs,theor_ref_sfs,theor_test_sfs,loglambda)
	return loglambda,theor_test_sfs,theor_ref_sfs

def re_param_neutral_alpha(x, ap_size, neutral_alpha, free_alpha, obs_test_sfs, obs_ref_sfs, inbreeding, mse):
	theta_site = x[ap_size]
	p_array = np.array([],dtype=np.float64)
	alpha = my_expand_alpha(x[0:ap_size])
	g_array = np.array([0],dtype=np.float64)
	d_array = np.array([0.5],dtype=np.float64)
	b_array = np.array([[0,0]])
	
	result = total_likelihood(obs_ref_sfs, obs_ref_sfs, theta_site, g_array, d_array, inbreeding, p_array, b_array, alpha, 0, mse)
	return -1*result[0]
    
def re_param(x, boundary_array, p_size, fixed_alpha, ap_size, neutral_alpha, free_alpha, obs_test_sfs, obs_ref_sfs, inbreeding, mse):
	theta_site = x[ap_size]
	p_array = x[(ap_size+1):(ap_size+1+p_size)]
	if (free_alpha and not(neutral_alpha)):
		alpha = my_expand_alpha(x[0:ap_size])
	else:
		alpha = my_expand_alpha(fixed_alpha)
	g_array = x[(ap_size+1+p_size):x.size]
	d_array = np.array([0.5]*g_array.size,dtype=np.float64)
	result = total_likelihood(obs_test_sfs, obs_ref_sfs, theta_site, g_array, d_array, inbreeding, p_array, boundary_array, alpha, 0, mse) 
	#print(alpha,x[6],x[7],x[8],x[9],result[0])
	return -1*result[0]
    
def re_param_neu(x, fixed_alpha, ap_size, neutral_alpha, free_alpha, obs_test_sfs, obs_ref_sfs, inbreeding, mse):
	theta_site = x[ap_size]
	p_array = np.array([],dtype=np.float64)
	if (free_alpha and not(neutral_alpha)):
		alpha = my_expand_alpha(x[0:ap_size])
	else:
		alpha = my_expand_alpha(fixed_alpha)
	g_array = np.array([0],dtype=np.float64)
	d_array = np.array([0.5],dtype=np.float64)
	b_array = np.array([[0,0]])
	result = total_likelihood(obs_test_sfs, obs_ref_sfs, theta_site, g_array, d_array, inbreeding, p_array, b_array, alpha, 0, mse)
	return -1*result[0]
    
def re_param_neu_lethal(x, fixed_alpha, ap_size, neutral_alpha, free_alpha, obs_test_sfs, obs_ref_sfs, inbreeding, mse):
	theta_site = x[ap_size]
	p_array = 1.0 - x[(x.size-1)]
	if (free_alpha and not(neutral_alpha)):
		alpha = my_expand_alpha(x[0:ap_size])
	else:
		alpha = my_expand_alpha(fixed_alpha)
	g_array = np.array([0],dtype=np.float64)
	d_array = np.array([0.5],dtype=np.float64)
	b_array = np.array([[0,0]])
	lethal_perc = x[x.size-1]
	result = total_likelihood(obs_test_sfs, obs_ref_sfs, theta_site, g_array, d_array, inbreeding, p_array, b_array, alpha, lethal_perc, mse)
	return -1*result[0]
    
def re_param_lethal(x, boundary_array, p_size, fixed_alpha, ap_size, neutral_alpha, free_alpha, obs_test_sfs, obs_ref_sfs, inbreeding, mse):
	theta_site = x[ap_size]
	p_array=x[(ap_size+1):(ap_size+1+p_size)]
	if (free_alpha and not(neutral_alpha)):
		alpha = my_expand_alpha(x[0:ap_size])
	else:
		alpha = my_expand_alpha(fixed_alpha)
	g_array = x[(ap_size+1+p_size):(x.size-1)]
	d_array = np.array([0.5]*g_array.size,dtype=np.float64)
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
	dominance = np.array([0.5]*initial_gamma_array.size,dtype=np.float64)
	mse = mse_gpu.MSE(n_samp,N_chrome,fold,zero_class)

	x_neu_alpha = np.append(init_alpha,initial_theta_site)
	maxfev = 100000*(x_neu_alpha.size+p_size+g_size)

	input_tuple = (boundary_array, p_size, init_alpha, ap_size, neutral_alpha, free_alpha, obs_test_sfs, obs_ref_sfs, inbreeding, mse)
	if(neutral_alpha and free_alpha):
		x_max_neu_alpha = scipy.optimize.minimize(re_param_neutral_alpha, x_neu_alpha, input_tuple[3:], method=optimizer, options={'maxfev': maxfev})
		for index,xa in enumerate(x_max_neu_alpha["x"][:ap_size]): init_alpha[index] = xa
		#print(x_max_neu_alpha)
	x = np.concatenate((np.ones(ap_size),np.array([initial_theta_site]),initial_p_array,initial_gamma_array)) 

	x_max = scipy.optimize.minimize(re_param, x, input_tuple, method=optimizer, options={'maxfev': maxfev})
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
	x_max_neu = scipy.optimize.minimize(re_param_neu, x_neu, input_tuple[2:], method=optimizer, options={'maxfev': maxfev}) 
	all_neutral_lik = -1*x_max_neu["fun"]
	max_theta_site_neu = x_max_neu["x"][ap_size]
    
	if(zero_class):
		x_neu_lethal = np.concatenate((np.ones(ap_size),np.array([initial_theta_site]),initial_lethal_array[0:1]))
		x_max_neu_lethal = scipy.optimize.minimize(re_param_neu_lethal, x_neu_lethal, input_tuple[2:], method=optimizer, options={'maxfev': maxfev})   
		max_neutral_lethal_lik = -1*x_max_neu_lethal["fun"]
		max_neu_lethal_perc = x_max_neu_lethal["x"][x_max_neu_lethal["x"].size-1]
		max_theta_site_neu_lethal = x_max_neu_lethal["x"][ap_size]
        
		x_lethal = np.concatenate((np.ones(ap_size),np.array([initial_theta_site]),initial_p_array,initial_gamma_array,initial_lethal_array[1:2]))  
		x_max_lethal = scipy.optimize.minimize(re_param_lethal, x_lethal, input_tuple, method=optimizer, options={'maxfev': maxfev})
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
    
	f.write(';[{} {}]'.format(boundary_array[0][0],boundary_array[0][1]))
	for boundary in boundary_array[1:]:
		f.write(",[{} {}]".format(boundary[0],boundary[1]))
    
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
			f.write('{:6.4f} {:6.4f}, '.format(max_gamma[index],p))
    	
	if (boundary_array[stop,0] == boundary_array[stop,1]):
		max_gamma[stop] = boundary_array[stop,0]
    
	if (zero_class):
		f.write('{:6.4f} {:6.4f}, theta {:6.4f}): '.format(max_gamma[stop],pfinal,max_theta_site))
	else:
		f.write('{:6.4f} {:6.4f}): '.format(max_gamma[stop],pfinal))
    
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
				f.write('{:6.4f} {:6.4f}, '.format(max_lethal_gamma[index], lethal_p))
		if (boundary_array[stop,0] == boundary_array[stop,1]):
			max_lethal_gamma[stop]=boundary_array[stop,0]
							
		f.write('{:6.4f} {:6.4f}, lethal {:6.4f}, theta {:6.4f}): {:6.4f}'.format(max_lethal_gamma[stop],pfinal,max_lethal_perc,max_theta_site_lethal,max_lethal_lik))
    
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

	my_test_sfs = np.loadtxt("sfs_test_out.txt", delimiter="\n", dtype=np.float64, unpack=False)
	my_ref_sfs = np.loadtxt("sfs_ref_out.txt", delimiter="\n", dtype=np.float64, unpack=False)
	
	file_name = "test_ml.txt"
	my_boundary_array = np.array([[0,0],[-40,-250]])
	initial_gamma_array = np.array([0,-50],dtype=np.float64)
	initial_p_array = np.array([0.8],dtype=np.float64)
	initial_theta_site = 0.035
	initial_lethal_array = np.array([0.01,0.01],dtype=np.float64)
	SFS_size = my_test_sfs.size
	my_expand_alpha = lambda x: expand_alpha(x, SFS_size, alpha_pattern)
	before = time.time()
	ml_results = maximum_likelihood(my_test_sfs, my_ref_sfs, file_name, initial_gamma_array, my_boundary_array, initial_p_array, initial_theta_site, initial_lethal_array, alpha_pattern, free_alpha, neutral_alpha, fold, zero_class, 4000, optimizer = 'Nelder-Mead')
	process_time = time.time() - before
	print(process_time,ml_results)
	# m_alpha=np.array([0.982322869711217,0.982197875401603,0.982197327162886,0.982196638455358,0.982195239794138,0.982205653372654],dtype=np.float64) for when Nchrome=4000,free=neutral=True
# 	#m_alpha_exp = my_expand_alpha(m_alpha)
# 	m_alpha_exp = ml_results[3]
# 	p_array = np.array([0.849887384658254],dtype=np.float64)
# 	g_array = np.array([0,-1.946871547510721e+02],dtype=np.float64)
# 	mse2 = mse_gpu.MSE(160,4000,fold,zero_class)
# 	results = total_likelihood(my_test_sfs, my_ref_sfs, 0.0102, g_array, [0.5,0.5], 0, p_array, my_boundary_array, m_alpha_exp, 0, mse2)
# 	print(results[0])
    
