/*!\file
* \brief proto-API for building site frequency spectra
*
* spectrum.h is a prototype API for accelerating site frequency spectrum analysis on the GPU.
* Though functions Spectrum::population_frequency_histogram and Spectrum::site_frequency_spectrum are accelerated on the GPU,
* the CUDA specific code is not in the spectrum.h header file and thus, like go_fish_data_struct.h, spectrum.h can be included
* in either CUDA C/C++ (*.cu) or standard C, C++ (*.c, *.cpp) source files.
*/
/*
 * spectrum.h
 *
 *      Author: David Lawrie
 */

#ifndef SPECTRUM_H_
#define SPECTRUM_H_

#include "../include/go_fish_data_struct.h"

///Namespace for site frequency spectrum data structure and functions (work in progress)
namespace Spectrum{

///site frequency spectrum data structure (at the moment, functions only generate SFS for a single population at a single time point)
struct SFS{
	float * frequency_spectrum; ///<site frequency spectrum data structure
	int * populations; ///<which populations are in SFS
	int * sample_size; ///<number of samples taken for each population
	int num_populations; ///<number of populations in SFS
	float num_sites;  ///<number of sites in SFS
	float num_mutations; ///<number of segregating mutations in SFS
	int sampled_generation; ///<number of generations in the simulation at time of sampling

	//!default constructor
	SFS();
	//!default destructor
	~SFS();
};

///frequency histogram of mutations at a single time point in a single population
void population_frequency_histogram(SFS & mySFS, const GO_Fish::allele_trajectories & all_results, const int sample_index, const int population_index, int cuda_device = -1);

///create a single-population SFS from a single time sample from an allele trajectory
void site_frequency_spectrum(SFS & mySFS, const GO_Fish::allele_trajectories & all_results, const int sample_index, const int population_index, const int sample_size, int cuda_device = -1);

} /*----- end namespace SPECTRUM ----- */

#endif /* SPECTRUM_H_ */
