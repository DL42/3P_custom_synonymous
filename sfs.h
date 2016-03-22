/*
 * sfs.h
 *
 *      Author: David Lawrie
 */

#ifndef SFS_H_
#define SFS_H_
#include <cuda_runtime.h>
#include "shared.h"

struct sfs{
	int * frequency_spectrum;
	int ** frequency_age_spectrum;
	int * populations; //which populations are in SFS
	int * num_samples; //number of samples taken for each population
	int num_populations;
	int num_sites;
	int total_generations; //number of generations in the simulation

	sfs();
	~sfs();
};

__host__ __forceinline__ sim_result sequencing_sample(sim_result sim, int * population, int * num_samples, const int seed);

//multi-population sfs
__host__ __forceinline__ sfs site_frequency_spectrum(sim_result sim, int * population, int * num_samples, int num_sfs_populations, const int seed);

//multi-time point, multi-population sfs
__host__ __forceinline__ sfs temporal_site_frequency_spectrum(sim_result sim, int * population, int * num_samples, int num_sfs_populations, const int seed);

//trace frequency trajectories of mutations from generation start to generation end in a (sub-)population
//can track an individual mutation or groups of mutation by specifying when the mutation was "born", in which population, with what threadID
__host__ __forceinline__ float ** trace_mutations(sim_result * sim, int generation_start, int generation_end, int population, int generation_born = -1, int population_born = -1, int threadID = -1);

#endif /* SFS_H_ */
