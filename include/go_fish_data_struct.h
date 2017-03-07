/*
 * go_fish_data_struct.h
 *
 *      Author: David Lawrie
 *      GO Fish data structures
 */

#ifndef GO_FISH_DATA_H_
#define GO_FISH_DATA_H_

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>

namespace SPECTRUM{ class transfer_allele_trajectories; }

namespace GO_Fish{

/* ----- sim result output ----- */
struct mutID{
	int origin_generation; //generation in which mutation appeared in simulation
	int origin_population; //population in which mutation first arose
	int origin_threadID; //threadID that generated mutation; if negative, flag to preserve mutation in simulation (not filter out if lost or fixed)
    int DFE_category; //discrete DFE category

    inline std::string toString();
};

struct allele_trajectories{
	//----- initialization parameters -----
	struct sim_constants{
		int seed1;
		int seed2;
		int num_generations;
		float num_sites;
		int num_populations;
		bool init_mse;
		int prev_sim_sample;
		int compact_interval;
		int device;

		sim_constants();
	};

	sim_constants sim_input_constants; //for initializing the next simulation
	//----- end -----

	allele_trajectories();

	//number of time samples taken during simulation run
	inline int num_time_samples();

	//number of reported mutations in the final time sample (maximal number of reported mutations in the allele_trajectories)
	inline int maximal_num_mutations();
	//number of reported mutations in the time sample index
	inline int num_mutations_time_sample(int sample_index);

	//final generation of simulation
	inline int final_generation();
	//generation of simulation in the time sample index
	inline int sampled_generation(int sample_index);

	//frequency of mutation at time sample sample_index, population population_index, mutation mutation_index
	//frequency of mutation before it is generated in the simulation will be reported as 0 (not an error)
	inline float frequency(int sample_index, int population_index, int mutation_index);

	//mutation IDs output from the simulation
	inline mutID mutation_ID(int mutation_index);

	inline void delete_time_sample(int sample_index);

	inline void free_memory();

	~allele_trajectories();

	template <typename Functor_mutation, typename Functor_demography, typename Functor_migration, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance, typename Functor_preserve, typename Functor_timesample>
	friend void run_sim(allele_trajectories & all_results, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_migration mig_prop, const Functor_selection sel_coeff, const Functor_inbreeding FI, const Functor_dominance dominance, const Functor_preserve preserve_mutations, const Functor_timesample take_sample, const allele_trajectories & prev_sim);

	friend class SPECTRUM::transfer_allele_trajectories;

private:

	struct time_sample{
		float * mutations_freq; //allele frequency of mutations in final generation
		mutID * mutations_ID; //unique ID consisting of generation, population, threadID, and device
		bool * extinct; //extinct[pop] == true, flag if population is extinct by time sample
		int * Nchrom_e; //effective number of chromosomes in each population
		int num_mutations; //number of mutations in array (array length for age/freq, columns)
		int sampled_generation; //number of generations in the simulation at point of sampling

		time_sample();
		~time_sample();
	};

	inline void initialize_run_constants();

	inline void initialize_sim_result_vector(int new_length);

	sim_constants sim_run_constants; //stores inputs of the simulation run currently held by time_samples
	time_sample ** time_samples; //the actual allele trajectories output from the simulation
	unsigned int length; //number of time samples taken from the simulation
};
/* ----- end sim result output ----- */

} /* ----- end namespace GO_Fish ----- */

/* ----- import inline function definitions ----- */
#include "../source/inline_go_fish_data_struct.hpp"
/* ----- end ----- */

#endif /* GO_FISH_DATA_H_ */
