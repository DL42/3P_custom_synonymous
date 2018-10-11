/*!\file
* \brief proto-API for building site frequency spectra (contains the titular namespace Spectrum)
*
* spectrum.h is a prototype API for accelerating site frequency spectrum analysis on the GPU.
* Though functions Spectrum::population_frequency_histogram and Spectrum::site_frequency_spectrum are accelerated on the GPU,
* the CUDA specific code is not in the spectrum.h header file and thus, like go_fish_data_struct.h, spectrum.h can be included
* in either CUDA (*.cu) or standard C, C++ (*.c, *.cpp) source files.
*/
/*
 * spectrum.h
 *
 *      Author: David Lawrie
 */

#ifndef SPECTRUM_H_
#define SPECTRUM_H_

#include "../3P/go_fish_data_struct.h"
#include "_internal/cuda_helper_fun.cuh"

///Namespace for site frequency spectrum data structure and functions. (in prototype-phase)
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

struct MSE{
	float * d_freq_pop_spectrum; ///<site frequency spectrum data structure (non-zero frequency, population)
	float * d_population_spectrum; ///<site frequency spectrum data structure (population, accumulated)
	float * d_binomial; //binomial distribution
	float * d_frequency_spectrum; ///<site frequency spectrum data structure (sampling)
	float * h_frequency_spectrum; ///<site frequency spectrum data structure (sampling)
	float * d_exp_snp_total; ///expected SNP total
	double * d_freq; ///temp vector used to fill up d_mse_integral
	double * d_mse_integral; ///integrated mse vector used to fill up d_freq_index
	
	void * d_temp_storage_integrate;
	size_t temp_storage_bytes_integrate;
	void * d_temp_storage_reduce;
	size_t temp_storage_bytes_reduce;
	
	
	float N_ind; ///number of individuals in mse calculation
	float F; ///inbreeding coefficient
	int Nchrom_e; ///population size of mse calculation
	int sample_size; ///<number of samples taken
	float num_sites;  ///<number of sites in SFS
	int cuda_device; ///cuda device to run on
	cudaStream_t stream; //cuda stream to run on
	
	///create a an array on a device, \p d_binomial_out, of size sample_size*(population_size-1)
	///when used in conjunction with void site_frequency_spectrum(SFS & mySFS, const SFS & inSFS, float * d_binomial, const int sample_size, int cuda_device = -1)
	///set cuda_device
	MSE(const int sample_size, int population_size, float inbreeding, int cuda_device = -1);
	//!default destructor
	~MSE();
};

///create a frequency histogram of mutations at a single time point \p sample_index in a single population \p population_index store in \p mySFS
void population_frequency_histogram(SFS & mySFS, const GO_Fish::allele_trajectories & all_results, const int sample_index, const int population_index, int cuda_device = -1);

///create a single-population SFS of size \p sample_size from a single time point \p sample_index in a single population \p population_index from allele trajectory \p all_results, store in \p mySFS
void site_frequency_spectrum(SFS & mySFS, const GO_Fish::allele_trajectories & all_results, const int sample_index, const int population_index, const int sample_size, int cuda_device = -1);

///create a single-population SFS from MSE, store in \p out
void site_frequency_spectrum(MSE & out);

} /*----- end namespace SPECTRUM ----- */

#endif /* SPECTRUM_H_ */
