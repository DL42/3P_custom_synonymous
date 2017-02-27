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

namespace SPECTRUM{ class transfer_allele_trajectories; }

namespace GO_Fish{

/* ----- sim result output ----- */
struct mutID{
	int origin_generation; //generation in which mutation appeared in simulation
	int origin_population; //population in which mutation first arose
	int origin_threadID; //threadID that generated mutation; if negative, flag to preserve mutation in simulation (not filter out if lost or fixed)
    int DFE_category; //discrete DFE category
};

struct allele_trajectories{
	//----- initialization parameters -----
	struct sim_input_constants{
		int seed1;
		int seed2;
		int num_generations;
		float num_sites;
		int num_discrete_DFE_categories;
		int num_populations;
		bool init_mse;
		int prev_sim_sample;
		int compact_rate;
		int device;

		sim_input_constants();
	}sim_input_constants;
	//----- end -----

	allele_trajectories();

	inline float frequency(int sample_index, int population_index, int mutation_index);

	//number of mutations in the final sample (maximal number of mutations in the allele_trajectories)
	inline int num_mutations();

	inline void delete_time_sample(int index);

	inline void free_memory();

	~allele_trajectories();

	template <typename Functor_mutation, typename Functor_demography, typename Functor_migration, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance, typename Functor_DFE, typename Functor_preserve, typename Functor_timesample>
	friend void run_sim(allele_trajectories & all_results, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_migration mig_prop, const Functor_selection sel_coeff, const Functor_inbreeding FI, const Functor_dominance dominance, const Functor_DFE discrete_DFE, const Functor_preserve preserve_mutations, const Functor_timesample take_sample, const allele_trajectories & prev_sim);

	friend class SPECTRUM::transfer_allele_trajectories;

private:

	struct time_sample{
		float * mutations_freq; //allele frequency of mutations in final generation
		mutID * mutations_ID; //unique ID consisting of generation, population, threadID, and device
		bool * extinct; //extinct[pop] == true, flag if population is extinct by end of simulation
		int * Nchrom_e; //effective number of chromosomes in each population
		int num_populations; //number of populations in freq array (array length, rows)
		int num_mutations; //number of mutations in array (array length for age/freq, columns)
		float num_sites; //number of sites in simulation
		int sampled_generation; //number of generations in the simulation at point of sampling

		time_sample();
		~time_sample();
	};

	inline void initialize_sim_result_vector(int new_length);

	time_sample ** time_samples;
	unsigned int length;
};
/* ----- end sim result output ----- */

} /* ----- end namespace GO_Fish ----- */

/* ----- import inline function definitions ----- */
#include "inline_go_fish_data_struct.hpp"
/* ----- end ----- */

#endif /* GO_FISH_DATA_H_ */
