/*
 * sfs.h
 *
 *      Author: David Lawrie
 */

#ifndef SFS_H_
#define SFS_H_

#include "go_fish_data_struct.h"

namespace SFS{

struct sfs{
	int * frequency_spectrum;
	int * populations; //which populations are in SFS
	int * num_samples; //number of samples taken for each population
	int num_populations;
	int num_sites;
	int sampled_generation; //number of generations in the simulation

	sfs();
	~sfs();
};

//single-population sfs, only segregating mutations
__host__  sfs site_frequency_spectrum(const GO_Fish::time_sample * const sim, int population, int cuda_device = -1);

/*
__host__ GO_Fish::time_sample sequencing_sample(GO_Fish::time_sample & sim, int * population, int * num_samples, const int seed);

//multi-time point, multi-population sfs
__host__ sfs temporal_site_frequency_spectrum(GO_Fish::time_sample & sim, int * population, int * num_samples, int num_sfs_populations, const int seed);

//trace frequency trajectories of mutations from generation start to generation end in a (sub-)population
//can track an individual mutation or groups of mutation by specifying when the mutation was "born", in which population, with what threadID
__host__ float ** trace_mutations(GO_Fish::time_sample * sim, int generation_start, int generation_end, int population, int generation_born = -1, int population_born = -1, int threadID = -1);
*/

} /*----- end namespace SFS ----- */

#endif /* SFS_H_ */
