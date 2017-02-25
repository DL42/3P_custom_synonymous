/*
 * shared.cu
 *
 *      Author: David Lawrie
 *      GO Fish data structures
 */

#include "go_fish_data_struct.h"
#include "shared.cuh"

namespace GO_Fish{

time_sample::time_sample(): num_populations(0), num_mutations(0), num_sites(0), sampled_generation(0) { mutations_freq = 0; mutations_ID = 0; extinct = 0; Nchrom_e = 0; /*set pointers to NULL*/}
time_sample::~time_sample(){
	if(mutations_freq){ cudaCheckErrors(cudaFreeHost(mutations_freq),-1,-1); mutations_freq = 0; }
	if(mutations_ID){ cudaCheckErrors(cudaFreeHost(mutations_ID),-1,-1); mutations_ID = 0; }
	if(extinct){ delete [] extinct; extinct = 0; }
	if(Nchrom_e){ delete [] Nchrom_e; Nchrom_e = 0; }
}

allele_trajectories::allele_trajectories(): length(0) { time_samples = 0; }
allele_trajectories::~allele_trajectories(){ free_memory(); }
allele_trajectories::sim_input_constants::sim_input_constants(): seed1(0xbeeff00d), seed2(0xdecafbad), num_generations(0), num_sites(1000), num_discrete_DFE_categories(1), num_populations(1), init_mse(true), prev_sim_sample(-1), compact_rate(35), device(-1) {}

}/* ----- end namespace GO_Fish ----- */
