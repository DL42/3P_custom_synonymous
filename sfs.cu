/*
 * sfs.cu
 *
 *      Author: David Lawrie
 */

#include "sfs.h"

sfs::sfs(): num_populations(0), num_sites(0), total_generations(0) {frequency_spectrum = NULL; frequency_age_spectrum = NULL; populations = NULL; num_samples = NULL;}
sfs::~sfs(){ if(frequency_spectrum){ delete[] frequency_spectrum; } if(frequency_age_spectrum){ delete[] frequency_age_spectrum; } if(populations){ delete[] populations; } if(num_samples){ delete[] num_samples; }}

__host__ __forceinline__ sim_result sequencing_sample(sim_result sim, int * population, int * num_samples, const int seed){
	//neutral neu;
	//no_mig mig;
	//migration_selection_drift(float * mutations_freq, const int mutations_Index, const int array_Length, const int N, const Functor_migration mig_prop, const Functor_selection sel_coeff, const float h, const float F, const int seed, const int population, const int num_populations, const int generation);
	//compact(sim_struct & mutations, const int generation, const Functor_mutation mu_rate, const Functor_demography demography, const float num_sites, const int compact_rate)
	return sim_result();
}

//multi-population sfs
__host__ __forceinline__ sfs site_frequency_spectrum(sim_result sim, int * population, int * num_samples, int num_sfs_populations, const int seed){
	sim_result samp = sequencing_sample(sim, population, num_samples, seed);

	return sfs();
}

//multi-time point, multi-population sfs
__host__ __forceinline__ sfs temporal_site_frequency_spectrum(sim_result sim, int * population, int * num_samples, int num_sfs_populations, const int seed){
	sim_result samp = sequencing_sample(sim, population, num_samples, seed);

	return sfs();
}

//trace frequency trajectories of mutations from generation start to generation end in a (sub-)population
//can track an individual mutation or groups of mutation by specifying when the mutation was "born", in which population, with what threadID
__host__ __forceinline__ float ** trace_mutations(sim_result * sim, int generation_start, int generation_end, int population, int generation_born /*= -1*/, int population_born /*= -1*/, int threadID /*= -1*/){

	return NULL;
}
