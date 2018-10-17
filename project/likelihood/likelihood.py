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
selected = mse_gpu.calc_sfs_mse(gamma,dominance,inbreeding,proportion,theta,alpha,mse)

def theor_neutral(theta, alpha, mse):
	gamma = np.array([0],dtype=np.float32)
	dominance = np.array([0.5],dtype=np.float32) #effectively ignored since inbreeding is set to 1
	proportion = np.array([1],dtype=np.float32)
	inbreeding = 0
	return mse_gpu.calc_sfs_mse(gamma,dominance,inbreeding,proportion,theta,alpha,mse)

neutral = theor_neutral(theta, alpha, mse)

# num_test_sites=sum(obs_test_sfs)
# num_ref_sites=sum(obs_ref_sfs)
# n_samp = obs_test_sfs.size - 1 
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

alpha = np.array([1]*79,dtype=np.float32)
boundary_array = np.array([[0.0,0.0],[0.0,-500]])
p_array = np.array([0.85])
sel = selected*num_sites
neu = neutral*num_sites
lik = total_likelihood(sel, neu, 0.01, gamma, dominance, inbreeding, p_array, boundary_array, alpha, 0, mse)
before = time.time()
lik = total_likelihood(sel, neu, 0.01, gamma, dominance, inbreeding, p_array, boundary_array, alpha, 0, mse)
process_time = time.time() - before
print(process_time)
print(lik[0])

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
	
SFS_size = 81
alpha_pattern = np.array([[0,1],[1,3],[3,7],[7,15],[15,31],[31,79]])
check_alpha_pattern(SFS_size, alpha_pattern)
alpha = expand_alpha(alpha_in, SFS_size, alpha_pattern)
print(alpha)

def output_results(file_name):

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
    f.write('{};'.format((1 - fsum(initial_p_array))))
    
    f.write('{:6.4f};'.format(initial_theta_site))
    f.write('{},{},{})\t('.format(initial_lethal_array[0],initial_lethal_array[1],N_chrome))
    
    stop = max_gamma.size - 1
    pfinal = 1
    if(stop != 0):
    	pfinal = 1 - fsum(max_p_array)
    	for gamma, boundary, p in zip(max_gamma[:stop], boundary_array[:stop], max_p_array):
        	if (boundary[0] == boundary[1]):
            	gamma = boundary[0]
        	f.write('{ } {:6.4f}, '.format(gamma,p))
    	
    if (boundary_array[stop,0] == boundary_array[stop,1]):
        max_gamma[stop] = boundary_array[stop,0]
    
    if (zero_class):
        f.write('{ } {:6.4f}, theta {:6.4f}): '.format(max_gamma[stop],pfinal,max_theta_site))
    else:
        f.write('{ } {:6.4f}): '.format(max_gamma[stop],pfinal))
    
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
			for lethal_gamma, boundary, lethal_p in zip(max_lethal_gamma[:stop], boundary_array[:stop], max_lethal_p_array):
				if (boundary_array[index,0] == boundary_array[index,1]):
					lethal_gamma = boundary[0]
				f.write('{ } {:6.4f}, '.format(lethal_gamma, lethal_p)
			if (boundary_array[stop,0] == boundary_array[stop,1]):
				max_lethal_gamma[stop]=boundary_array[stop,0]
							
        f.write('{ } {:6.4f}, lethal {:6.4f}, theta {:6.4f}): {:6.4f}'.format(max_lethal_gamma(stop),pfinal,max_lethal_perc,max_theta_site_lethal,max_lethal_lik))
    
    f.write('\n')
    fclose(fileID)

if __name__ == '__main__':
    pass
    