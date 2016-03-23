/*
 * shared.cu
 *
 *      Author: David Lawrie
 *      for structures and functions used by both go_fish and by sfs
 */

#include "shared.cuh"

sim_result::sim_result(): num_populations(0), num_mutations(0), num_sites(0), total_generations(0) { mutations_freq = NULL; mutations_ID = NULL; extinct = NULL; }
sim_result::~sim_result(){ if(mutations_freq){ cudaCheckErrors(cudaFreeHost(mutations_freq),-1,-1); } if(mutations_ID){ cudaCheckErrors(cudaFreeHost(mutations_ID),-1,-1); } if(extinct){ delete [] extinct; } }

__device__ __noinline__ int Rand1(unsigned int i, float mean, float var, float N){
	if(mean <= 6){ return poiscdfinv(uint_float_01(i), mean); }
	else if(mean >= N-6){ return N - poiscdfinv(uint_float_01(i), N-mean); } //flip side of binomial, when 1-p is small
	return round(normcdfinv(uint_float_01(i))*sqrtf(var)+mean);
}
