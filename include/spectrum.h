/*
 * spectrum.h
 *
 *      Author: David Lawrie
 */

#ifndef SPECTRUM_H_
#define SPECTRUM_H_

#include "../include/go_fish_data_struct.h"

namespace SPECTRUM{

struct sfs{
	double * frequency_spectrum;
	int * populations; //which populations are in SFS
	int * sample_size; //number of samples taken for each population
	int num_populations;
	float num_sites;
	double num_mutations;
	int sampled_generation; //number of generations in the simulation at time of sampling

	sfs();
	~sfs();
};

//single-population sfs
sfs site_frequency_spectrum(const GO_Fish::allele_trajectories & all_results, const int sample_index, const int population_index, const unsigned int sample_size = 0, int cuda_device = -1);

} /*----- end namespace SPECTRUM ----- */

#endif /* SPECTRUM_H_ */
