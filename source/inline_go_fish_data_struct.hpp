/*
 * inline_go_fish_data_struct.hpp
 *
 *      Author: David Lawrie
 *      inline function definitions for GO Fish data structures
 */

#ifndef INLINE_GOFISH_DATA_FUNCTIONS_HPP_
#define INLINE_GOFISH_DATA_FUNCTIONS_HPP_

namespace GO_Fish{

template<typename T>
std::string tostring(const T& value)
{
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

inline std::string mutID::toString(){ return "("+tostring(origin_generation)+","+tostring(origin_population)+","+tostring(abs(origin_threadID))+","+tostring(DFE_category)+")"; } //abs(origin_threadID) so the user doesn't get confused by the preservation flag on ID, here too for eventual allele trajectory.toString() or toFile() more likely

inline int allele_trajectories::num_time_samples(){ return length; }

//number of reported mutations in the final time sample (maximal number of reported mutations in the allele_trajectories)
inline int allele_trajectories::maximal_num_mutations(){ return num_mutations_time_sample(length-1); }

//number of reported mutations in the time sample index
inline int allele_trajectories::num_mutations_time_sample(int sample_index){
	if(!time_samples || length == 0){ fprintf(stderr,"num_mutations error: empty allele_trajectories\n"); exit(1); }
	if(sample_index < 0 || sample_index > length){ fprintf(stderr,"num_mutations error: requested sample index out of bounds: sample %d [0 %d)\n",sample_index,length); exit(1); }
	return time_samples[sample_index]->num_mutations;
}

//final generation of simulation
inline int allele_trajectories::final_generation(){ return sampled_generation(length-1); }

//generation of simulation in the time sample index
inline int allele_trajectories::sampled_generation(int sample_index){
	if(!time_samples || length == 0){ fprintf(stderr,"sampled_generation error: empty allele_trajectories\n"); exit(1); }
	if(sample_index < 0 || sample_index > length){ fprintf(stderr,"sampled_generation error: requested sample index out of bounds: sample %d [0 %d)\n",sample_index,length); exit(1); }
	return time_samples[sample_index]->sampled_generation;
}

//frequency of mutation at time sample sample_index, population population_index, mutation mutation_index
//frequency of mutation before it is generated in the simulation will be reported as 0 (not an error)
inline float allele_trajectories::frequency(int sample_index, int population_index, int mutation_index){
	int num_populations = sim_input_constants.num_populations;
	int num_mutations;
	if((sample_index >= 0 && sample_index < length) && (population_index >= 0 && population_index < num_populations) && (mutation_index >= 0 && mutation_index < time_samples[length-1]->num_mutations)){
		num_mutations = time_samples[length-1]->num_mutations;
		int num_mutations_in_sample = time_samples[sample_index]->num_mutations;
		if(mutation_index >= num_mutations_in_sample){ return 0; }
		return time_samples[sample_index]->mutations_freq[mutation_index+population_index*num_mutations_in_sample];
	}
	else{
		if(!time_samples || length == 0){ fprintf(stderr,"frequency error: empty allele_trajectories\n"); exit(1); }
		num_mutations = time_samples[length-1]->num_mutations;
		fprintf(stderr,"frequency error: index out of bounds: sample %d [0 %d), population %d [0 %d), mutation %d [0 %d)\n",sample_index,length,population_index,num_populations,mutation_index,num_mutations); exit(1);
	}
}

//mutation IDs output from the simulation
inline mutID allele_trajectories::mutation_ID(int mutation_index){
	if(length > 0 && time_samples){
		if(mutation_index >= 0 && mutation_index < time_samples[length-1]->num_mutations){
			mutID temp;
			temp.origin_generation = time_samples[length-1]->mutations_ID[mutation_index].origin_generation;
			temp.origin_population = time_samples[length-1]->mutations_ID[mutation_index].origin_population;
			temp.origin_threadID = abs(time_samples[length-1]->mutations_ID[mutation_index].origin_threadID); //so the user doesn't get confused by the preservation flag on ID
			temp.DFE_category = time_samples[length-1]->mutations_ID[mutation_index].DFE_category;
			return temp;
		}
		fprintf(stderr,"mutation_ID error: requested mutation index out of bounds: mutation %d [0 %d)\n",mutation_index,maximal_num_mutations()); exit(1);
	}else{
		fprintf(stderr,"mutation_ID error: empty allele_trajectories\n"); exit(1);
	}
}

inline void allele_trajectories::delete_time_sample(int sample_index){
	if(sample_index >= 0 && sample_index < length){
		delete time_samples[sample_index];
		time_sample ** temp = new time_sample * [length-1];
		for(int i = 0; i < length; i++){
			if(i < sample_index){ temp[i] = time_samples[i]; }
			else if (i > sample_index){ temp[i-1] = time_samples[i]; }
		}
		time_samples = temp;
		length -= 1;
	}else{
		if(!time_samples || length == 0){ fprintf(stderr,"delete_time_sample error: empty allele_trajectories\n"); exit(1); }
		fprintf(stderr,"delete_time_sample error: requested sample index out of bounds: sample %d [0 %d)\n",sample_index,length); exit(1);
	}
}

inline void allele_trajectories::free_memory(){
	if(time_samples){
		for(int i = 0; i < length; i++){ delete time_samples[i]; }
		delete [] time_samples;
	}
	time_samples = NULL; length = 0;
}

inline void allele_trajectories::sim_constants::initialize(sim_constants & input){
	seed1 = input.seed1;
	seed2 = input.seed2;
	num_generations = input.num_generations;
	num_sites = input.num_sites;
	num_populations = input.num_populations;
	init_mse = input.init_mse;
	prev_sim_sample = input.prev_sim_sample;
	compact_interval = input.compact_interval;
	device = input.device;
}


inline void allele_trajectories::initialize_sim_result_vector(int new_length){
	free_memory(); //overwrite old data if any
	length = new_length;
	time_samples = new time_sample *[length];
	for(int i = 0; i < length; i++){ time_samples[i] = new time_sample(); }
	sim_run_constants.initialize(sim_input_constants);
}

} /* ----- end namespace GO_Fish ----- */

#endif /* INLINE_GOFISH_DATA_FUNCTIONS_HPP_ */
