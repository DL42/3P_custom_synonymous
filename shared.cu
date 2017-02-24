/*
 * shared.cu
 *
 *      Author: David Lawrie
 *      for structures and functions used by both go_fish and by sfs
 */

#include "shared.cuh"

GO_Fish::time_sample::time_sample(): num_populations(0), num_mutations(0), num_sites(0), sampled_generation(0) { mutations_freq = 0; mutations_ID = 0; extinct = 0; Nchrom_e = 0; /*set pointers to NULL*/}
GO_Fish::time_sample::~time_sample(){
	if(mutations_freq){ cudaCheckErrors(cudaFreeHost(mutations_freq),-1,-1); mutations_freq = 0; }
	if(mutations_ID){ cudaCheckErrors(cudaFreeHost(mutations_ID),-1,-1); mutations_ID = 0; }
	if(extinct){ delete [] extinct; extinct = 0; }
	if(Nchrom_e){ delete [] Nchrom_e; Nchrom_e = 0; }
}

GO_Fish::allele_trajectories::allele_trajectories(): length(0) { time_samples = 0; }
GO_Fish::allele_trajectories::~allele_trajectories(){ if(time_samples){ delete [] time_samples; time_samples = 0; } }
GO_Fish::allele_trajectories::sim_input_params::sim_input_params(): seed1(0xbeeff00d), seed2(0xdecafbad), num_generations(0), num_sites(1000), num_discrete_DFE_categories(1), num_populations(1), init_mse(true), prev_sim(time_sample()), compact_rate(35), device(-1) {}
GO_Fish::allele_trajectories::sim_input_params::~sim_input_params(){ }

__device__ int RNG::ApproxRandBinomHelper(unsigned int i, float mean, float var, float N){
	if(mean <= MEAN_BOUNDARY){ return poiscdfinv(uint_float_01(i), mean); }
	else if(mean >= N-MEAN_BOUNDARY){ return N - poiscdfinv(uint_float_01(i), N-mean); } //flip side of binomial, when 1-p is small
	return round(normcdfinv(uint_float_01(i))*sqrtf(var)+mean);
}
