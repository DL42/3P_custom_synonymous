/*
 * spectrum.h
 *
 *      Author: David Lawrie
 */

#ifndef SPECTRUM_H_
#define SPECTRUM_H_

#include "../include/go_fish_data_struct.h"

namespace SPECTRUM{

struct SFS{
	float * frequency_spectrum;
	int * populations; //which populations are in SFS
	int * sample_size; //number of samples taken for each population
	int num_populations;
	float num_sites;
	float num_mutations;
	int sampled_generation; //number of generations in the simulation at time of sampling

	SFS();
	~SFS();
};

//frequency histogram of mutations at a single time point in a single population
void population_frequency_histogram(SFS & mySFS, const GO_Fish::allele_trajectories & all_results, const int sample_index, const int population_index, int cuda_device = -1);

//single-population SFS
void site_frequency_spectrum(SFS & mySFS, const GO_Fish::allele_trajectories & all_results, const int sample_index, const int population_index, const int sample_size, int cuda_device = -1);

} /*----- end namespace SPECTRUM ----- */

#endif /* SPECTRUM_H_ */
