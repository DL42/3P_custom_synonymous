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
    int reserved; //reserved for later use

    //returns a string constant of mutID
    inline std::string toString();
};

struct allele_trajectories{
	//----- initialization parameters -----
	struct sim_constants{
		int seed1; //random number seed 1 of 2
		int seed2; //random number seed 2 of 2
		int num_generations; //number of generations in simulation
		float num_sites; //number of sites in simulation
		int num_populations; //number of populations in simulation
		bool init_mse; //true: initialize simulation in mutation_selection_equilibrium; false: initialize blank simulation or using previous simulation time sample
		int prev_sim_sample; //time sample of previous simulation to use for initializing current simulation, overridden by init_mse if init_mse = true
		int compact_interval; //how often to compact the simulation and remove fixed or lost mutations
		int device; //GPU identity to run simulation on, if -1 next available GPU will be assigned

		inline sim_constants();
	};

	sim_constants sim_input_constants; //for initializing the next simulation
	//----- end -----

	//default constructor
	inline allele_trajectories();

	//copy constructor
	inline allele_trajectories(const allele_trajectories & in);

	//copy assignment
	inline allele_trajectories & operator= (allele_trajectories in);

	//returns sim_constants of the simulation currently held by allele_trajectories
	inline sim_constants last_run_constants();

	//returns number of time samples taken during simulation run
	inline int num_time_samples();

	//returns number of reported mutations in the final time sample (maximal number of stored mutations in the allele_trajectories)
	inline int maximal_num_mutations();

	//returns number of segregating mutations reported by the simulation in the time sample index
	inline int num_mutations_time_sample(int sample_index);

	//returns final generation of simulation
	inline int final_generation();

	//return generation of simulation in the time sample index
	inline int sampled_generation(int sample_index);

	//returns whether or not population population_index is extinct in time sample sample_index
	inline bool extinct(int sample_index, int population_index);

	//returns the effective number of chromosomes of population_index in time sample sample_index
	inline int effective_number_of_chromosomes(int sample_index, int population_index);

	//returns the frequency of the mutation at time sample sample_index, population population_index, mutation mutation_index
	//frequency of mutation before it is generated in the simulation will be reported as 0 (not an error)
	inline float frequency(int sample_index, int population_index, int mutation_index);

	//returns the mutation ID at mutation_index
	inline mutID mutation_ID(int mutation_index);

	//deletes a single time sample
	//does not delete mutations_ID but does move the apparent length of the array, all_mutations, to the number of mutations in the next last time sample if the final time sample is deleted
	//if deleting the last time sample left in allele trajectories, will free all memory including mutations_ID
	inline void delete_time_sample(int sample_index);

	//deletes all memory held by allele_trajectories, resets constants to default
	inline void reset();

	//destructor
	inline ~allele_trajectories();

	template <typename Functor_mutation, typename Functor_demography, typename Functor_migration, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance, typename Functor_preserve, typename Functor_timesample>
	friend void run_sim(allele_trajectories & all_results, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_migration mig_prop, const Functor_selection sel_coeff, const Functor_inbreeding FI, const Functor_dominance dominance, const Functor_preserve preserve_mutations, const Functor_timesample take_sample, const allele_trajectories & prev_sim);

	friend void swap(allele_trajectories & a, allele_trajectories & b);

	friend class SPECTRUM::transfer_allele_trajectories;

private:

	struct time_sample{
		float * mutations_freq; //allele frequency of mutations in final generation
		bool * extinct; //extinct[pop] == true, flag if population is extinct by time sample
		int * Nchrom_e; //effective number of chromosomes in each population
		int num_mutations; //number of mutations in frequency array (columns array length for freq)
		int sampled_generation; //number of generations in the simulation at point of sampling

		time_sample();
		~time_sample();
	};

	inline void initialize_run_constants();

	inline void initialize_sim_result_vector(int new_length);

	sim_constants sim_run_constants; //stores inputs of the simulation run currently held by time_samples
	time_sample ** time_samples; //the actual allele trajectories output from the simulation
	int num_samples; //number of time samples taken from the simulation
	mutID * mutations_ID; //unique ID for each mutation in simulation
	int all_mutations; //number of mutations in mutation ID array - maximal set of mutations stored in allele_trajectories
};
/* ----- end sim result output ----- */

} /* ----- end namespace GO_Fish ----- */

/* ----- import inline function definitions ----- */
#include "../source/inline_go_fish_data_struct.hpp"
/* ----- end ----- */

#endif /* GO_FISH_DATA_H_ */
