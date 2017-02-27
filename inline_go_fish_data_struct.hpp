/*
 * inline_go_fish_data_struct.hpp
 *
 *      Author: David Lawrie
 *      inline function definitions for GO Fish data structures
 */

#ifndef INLINE_GOFISH_DATA_FUNCTIONS_HPP_
#define INLINE_GOFISH_DATA_FUNCTIONS_HPP_

namespace GO_Fish{

inline float allele_trajectories::frequency(int sample_index, int population_index, int mutation_index){
	int num_populations = sim_input_constants.num_populations;
	int num_mutations;
	if((sample_index >= 0 && sample_index < length) && time_samples && (population_index >= 0 && population_index < num_populations) && (mutation_index >= 0 && mutation_index < time_samples[length-1]->num_mutations)){
		num_mutations = time_samples[length-1]->num_mutations;
		int num_mutations_in_sample = time_samples[sample_index]->num_mutations;
		if(mutation_index >= num_mutations_in_sample){ return 0; }
		return time_samples[sample_index]->mutations_freq[mutation_index+population_index*num_mutations_in_sample];
	}
	else{
		if(!time_samples){ fprintf(stderr,"frequency error: empty allele_trajectories\n"); exit(1); }
		num_mutations = time_samples[length-1]->num_mutations;
		fprintf(stderr,"frequency error: index out of bounds: sample %d\t[0 %d), population %d\t[0 %d), mutation %d\t[0 %d)\n",sample_index,length,population_index,num_populations,mutation_index,num_mutations); exit(1);
	}
}

//number of mutations in the final sample (maximal number of mutations in the allele_trajectories)
inline int allele_trajectories::num_mutations(){
	if(!time_samples){ fprintf(stderr,"num_mutations(): empty allele_trajectories\n"); exit(1); }
	return time_samples[length-1]->num_mutations;
}

inline void allele_trajectories::delete_time_sample(int index){
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
		if(!time_samples){ fprintf(stderr,"delete_time_sample error: empty allele_trajectories\n"); exit(1); }
		fprintf(stderr,"delete_time_sample error: requested sample index out of bounds: sample %d\t[0 %d)\n",index,length); exit(1);
	}
}

inline void allele_trajectories::free_memory(){ if(time_samples){ delete [] time_samples; } time_samples = 0; length = 0; }

inline void allele_trajectories::initialize_sim_result_vector(int new_length){
	free_memory(); //overwrite old data if any
	length = new_length;
	time_samples = new time_sample *[length];
	for(int i = 0; i < length; i++){ time_samples[i] = new time_sample(); }
}

} /* ----- end namespace GO_Fish ----- */

#endif /* INLINE_GOFISH_DATA_FUNCTIONS_HPP_ */
