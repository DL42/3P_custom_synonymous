/*
 * shared.cu
 *
 *      Author: David Lawrie
 *      for structures and functions used by both go_fish and by sfs
 */

#include "shared.cuh"


GO_Fish::time_sample::time_sample(): num_populations(0), num_mutations(0), num_sites(0), sampled_generation(0) { mutations_freq = NULL; mutations_ID = NULL; extinct = NULL; Nchrom_e = NULL; }
GO_Fish::time_sample::~time_sample(){
	if(mutations_freq){ cudaCheckErrors(cudaFreeHost(mutations_freq),-1,-1); }
	if(mutations_ID){ cudaCheckErrors(cudaFreeHost(mutations_ID),-1,-1); }
	if(extinct){ delete [] extinct; }
	if(Nchrom_e){ delete [] Nchrom_e; }
}

GO_Fish::sim_result_vector::sim_result_vector(): length(0), seed1(0xbeeff00d), seed2(0xdecafbad), num_generations(0), num_sites(1000), num_populations(1), init_mse(true), prev_sim(time_sample()), compact_rate(35), device(-1) { time_samples = NULL; }
GO_Fish::sim_result_vector::~sim_result_vector(){ if(time_samples){ delete [] time_samples; } }

__device__ int RNG::ApproxRandBinomHelper(unsigned int i, float mean, float var, float N){
	if(mean <= MEAN_BOUNDARY){ return poiscdfinv(uint_float_01(i), mean); }
	else if(mean >= N-MEAN_BOUNDARY){ return N - poiscdfinv(uint_float_01(i), N-mean); } //flip side of binomial, when 1-p is small
	return round(normcdfinv(uint_float_01(i))*sqrtf(var)+mean);
}
