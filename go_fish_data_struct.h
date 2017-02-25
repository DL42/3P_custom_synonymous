/*
 * shared.cu
 *
 *      Author: David Lawrie
 *      GO Fish data structures
 */
#ifndef GO_FISH_DATA_H_
#define GO_FISH_DATA_H_

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

namespace GO_Fish{

/* ----- sim result output ----- */
struct mutID{
	int origin_generation; //generation in which mutation appeared in simulation
	int origin_population; //population in which mutation first arose
	int origin_threadID; //threadID that generated mutation; if negative, flag to preserve mutation in simulation (not filter out if lost or fixed)
    int DFE_category; //discrete DFE category
};

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

struct allele_trajectories{
	time_sample ** time_samples;
	unsigned int length;

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
	inline void free_memory(){ if(time_samples){ delete [] time_samples; } time_samples = 0; length = 0; }
	inline void delete_time_sample(int index){
		if(index >= 0 && index < length){
			delete time_samples[index];
			time_sample ** temp = new time_sample * [length-1];
			for(int i = 0; i < length; i++){
				if(i < index){ temp[i] = time_samples[i]; }
				else if (i > index){ temp[i-1] = time_samples[i]; }
			}
			time_samples = temp;
			length -= 1;
		}else{
			if(!time_samples){ fprintf(stderr,"delete_time_sample: empty allele_trajectories\n"); exit(1); }
			fprintf(stderr,"delete_time_sample: requested sample index out of bounds: sample %d\t[0\t %d)\n",index,length); exit(1);
		}
	}

	inline float frequency(int sample_index, int population_index, int mutation_index){
		int num_mutations = time_samples[length-1]->num_mutations;
		int num_populations = sim_input_constants.num_populations;
		if((sample_index >= 0 && sample_index < length) && (population_index >= 0 && population_index < num_populations) && (mutation_index >= 0 && mutation_index < num_mutations)){
			int num_mutations_in_sample = time_samples[sample_index]->num_mutations;
			if(mutation_index >= num_mutations_in_sample){ return 0; }
			return time_samples[sample_index]->mutations_freq[mutation_index+population_index*num_mutations_in_sample];
		}
		else{
			if(!time_samples){ fprintf(stderr,"frequencies: empty allele_trajectories\n"); exit(1); }
			fprintf(stderr,"frequencies: index out of bounds: sample %d\t[0\t %d), population %d\t[0\t %d), mutation %d\t[0\t %d)\n",sample_index,length,population_index,num_populations,mutation_index,num_mutations); exit(1);
		}
	}

	//template <typename Functor_mutation, typename Functor_demography, typename Functor_migration, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance, typename Functor_DFE, typename Functor_preserve, typename Functor_timesample>
	//friend void run_sim(allele_trajectories & all_results, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_migration mig_prop, const Functor_selection sel_coeff, const Functor_inbreeding FI, const Functor_dominance dominance, const Functor_DFE discrete_DFE, const Functor_preserve preserve_mutations, const Functor_timesample take_sample, const time_sample & prev_sim);

	inline time_sample* operator[](int index){
		if(index >= 0 && index < length){ return time_samples[index]; }
		else{ fprintf(stderr,"requested sample index out of bounds: sample %d\t[0\t %d)\n",index,length); exit(1); }
	}
	~allele_trajectories();
};
/* ----- end sim result output ----- */

} /* ----- end namespace GO_Fish ----- */

#endif /* GO_FISH_DATA_H_ */
