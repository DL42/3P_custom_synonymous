/*
 * spectrum.h
 *
 *      Author: David Lawrie
 */

#ifndef SPECTRUM_H_
#define SPECTRUM_H_

#include "go_fish_data_struct.h"

namespace SPECTRUM{

struct sfs{
	float * frequency_spectrum;
	int * populations; //which populations are in SFS
	int * num_samples; //number of samples taken for each population
	int num_populations;
	int num_sites;
	int sampled_generation; //number of generations in the simulation

	sfs();
	~sfs();
};

//single-population sfs
sfs site_frequency_spectrum(const GO_Fish::allele_trajectories & all_results, int sample_index, int population_index, unsigned int num_sequencing_samples = 0, int cuda_device = -1);

} /*----- end namespace SPECTRUM ----- */

#endif /* SPECTRUM_H_ */
