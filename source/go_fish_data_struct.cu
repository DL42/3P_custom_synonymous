/*
 * go_fish_data_struct.cu
 *
 *      Author: David Lawrie
 *      GO Fish data structures
 */

#include "../include/go_fish_data_struct.h"
#include "../source/shared.cuh"

namespace GO_Fish{

allele_trajectories::time_sample::time_sample(): num_mutations(0), sampled_generation(0) { mutations_freq = NULL; extinct = NULL; Nchrom_e = NULL; /*set pointers to NULL*/}
allele_trajectories::time_sample::~time_sample(){
	if(mutations_freq){ cudaCheckErrors(cudaFreeHost(mutations_freq),-1,-1); mutations_freq = NULL; }
	if(extinct){ delete [] extinct; extinct = NULL; }
	if(Nchrom_e){ delete [] Nchrom_e; Nchrom_e = NULL; }
}

void allele_trajectories::free_memory(){
	if(time_samples){
		for(int i = 0; i < num_samples; i++){ delete time_samples[i]; }
		delete [] time_samples;
	}
	if(mutations_ID){ cudaCheckErrors(cudaFreeHost(mutations_ID),-1,-1); }
	time_samples = NULL; num_samples = 0; mutations_ID = NULL; all_mutations = 0;
}

allele_trajectories::allele_trajectories(): num_samples(0), all_mutations(0) { time_samples = NULL; mutations_ID = NULL; }
allele_trajectories::~allele_trajectories(){ free_memory(); }
allele_trajectories::sim_constants::sim_constants(): seed1(0xbeeff00d), seed2(0xdecafbad), num_generations(0), num_sites(1000), num_populations(1), init_mse(true), prev_sim_sample(-1), compact_interval(35), device(-1) {}

}/* ----- end namespace GO_Fish ----- */
