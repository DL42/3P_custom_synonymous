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

inline std::string mutID::toString(){ return "("+tostring(origin_generation)+","+tostring(origin_population)+","+tostring(abs(origin_threadID))+","+tostring(reserved)+")"; } //abs(origin_threadID) so the user doesn't get confused by the preservation flag on ID, here too for eventual allele trajectory.toString() or toFile() more likely

inline allele_trajectories::sim_constants::sim_constants(): seed1(0xbeeff00d), seed2(0xdecafbad), num_generations(0), num_sites(1000), num_populations(1), init_mse(true), prev_sim_sample(-1), compact_interval(35), device(-1) {}

inline allele_trajectories::time_sample::time_sample(): num_mutations(0), sampled_generation(0) { mutations_freq = NULL; extinct = NULL; Nchrom_e = NULL; /*set pointers to NULL*/}
inline allele_trajectories::time_sample::~time_sample(){
	if(mutations_freq){ delete [] mutations_freq; mutations_freq = NULL; }
	if(extinct){ delete [] extinct; extinct = NULL; }
	if(Nchrom_e){ delete [] Nchrom_e; Nchrom_e = NULL; }
}

inline allele_trajectories::allele_trajectories(): num_samples(0), all_mutations(0) { time_samples = NULL; mutations_ID = NULL; }

inline allele_trajectories::allele_trajectories(const allele_trajectories & in){
	//can replace with shared pointers when moving to CUDA 7+ and C++11
	sim_input_constants = in.sim_input_constants;
	sim_run_constants = in.sim_run_constants;
	num_samples = in.num_samples;
	all_mutations = in.all_mutations;

	if(all_mutations > 0){
		mutations_ID = new mutID[all_mutations];
		memcpy(mutations_ID, in.mutations_ID, all_mutations*sizeof(mutID));
	}
	else{ mutations_ID = NULL; }

	if(num_samples > 0){
		time_samples = new time_sample*[num_samples];
		for(int i = 0; i < num_samples; i++){
			time_samples[i] = new time_sample;
			int num_mutations = in.time_samples[i]->num_mutations;
			time_samples[i]->num_mutations = num_mutations;
			time_samples[i]->sampled_generation = in.time_samples[i]->sampled_generation;
			int num_populations = sim_run_constants.num_populations;
			time_samples[i]->Nchrom_e = new int[num_populations];
			time_samples[i]->extinct = new bool[num_populations];
			for(int j = 0; j < num_populations; j++){
				time_samples[i]->Nchrom_e[j] = in.time_samples[i]->Nchrom_e[j];
				time_samples[i]->extinct[j] = in.time_samples[i]->extinct[j];
			}
			time_samples[i]->mutations_freq = new float[num_mutations];
			memcpy(time_samples[i]->mutations_freq,in.time_samples[i]->mutations_freq,num_mutations*sizeof(float));
		}
	}
	else{ time_samples = NULL; }
}

inline allele_trajectories & allele_trajectories::operator= (allele_trajectories in){
	swap(*this, in);
	return *this;
}

inline allele_trajectories::sim_constants allele_trajectories::last_run_constants(){ return sim_run_constants; }

inline int allele_trajectories::num_time_samples(){ return num_samples; }

inline int allele_trajectories::maximal_num_mutations(){ return all_mutations; }

inline int allele_trajectories::num_mutations_time_sample(int sample_index){
	if(!time_samples || num_samples == 0){ fprintf(stderr,"num_mutations error: empty allele_trajectories\n"); exit(1); }
	if(sample_index < 0 || sample_index > num_samples){ fprintf(stderr,"num_mutations error: requested sample index out of bounds: sample %d [0 %d)\n",sample_index,num_samples); exit(1); }
	return time_samples[sample_index]->num_mutations;
}

inline int allele_trajectories::final_generation(){ return sampled_generation(num_samples-1); }

inline int allele_trajectories::sampled_generation(int sample_index){
	if(!time_samples || num_samples == 0){ fprintf(stderr,"sampled_generation error: empty allele_trajectories\n"); exit(1); }
	if(sample_index < 0 || sample_index > num_samples){ fprintf(stderr,"sampled_generation error: requested sample index out of bounds: sample %d [0 %d)\n",sample_index,num_samples); exit(1); }
	return time_samples[sample_index]->sampled_generation;
}

inline bool allele_trajectories::extinct(int sample_index, int population_index){
	if(!time_samples || num_samples == 0){ fprintf(stderr,"extinct error: empty allele_trajectories\n"); exit(1); }
	if(sample_index < 0 || sample_index > num_samples){ fprintf(stderr,"extinct error: requested sample index out of bounds: sample %d [0 %d)\n",sample_index,num_samples); exit(1); }
	return time_samples[sample_index]->extinct[population_index];
}

inline int allele_trajectories::effective_number_of_chromosomes(int sample_index, int population_index){
	if(!time_samples || num_samples == 0){ fprintf(stderr,"population_size error: empty allele_trajectories\n"); exit(1); }
	if(sample_index < 0 || sample_index > num_samples){ fprintf(stderr,"population_size error: requested sample index out of bounds: sample %d [0 %d)\n",sample_index,num_samples); exit(1); }
	return time_samples[sample_index]->Nchrom_e[population_index];
}

inline float allele_trajectories::frequency(int sample_index, int population_index, int mutation_index){
	int num_populations = sim_input_constants.num_populations;
	int num_mutations;
	if((sample_index >= 0 && sample_index < num_samples) && (population_index >= 0 && population_index < num_populations) && (mutation_index >= 0 && mutation_index < time_samples[num_samples-1]->num_mutations)){
		num_mutations = time_samples[num_samples-1]->num_mutations;
		int num_mutations_in_sample = time_samples[sample_index]->num_mutations;
		if(mutation_index >= num_mutations_in_sample){ return 0; }
		return time_samples[sample_index]->mutations_freq[mutation_index+population_index*num_mutations_in_sample];
	}
	else{
		if(!time_samples || num_samples == 0){ fprintf(stderr,"frequency error: empty allele_trajectories\n"); exit(1); }
		num_mutations = time_samples[num_samples-1]->num_mutations;
		fprintf(stderr,"frequency error: index out of bounds: sample %d [0 %d), population %d [0 %d), mutation %d [0 %d)\n",sample_index,num_samples,population_index,num_populations,mutation_index,num_mutations); exit(1);
	}
}

inline mutID allele_trajectories::mutation_ID(int mutation_index){
	if(num_samples > 0 && time_samples){
		if(mutation_index >= 0 && mutation_index < all_mutations){
			mutID temp;
			temp.origin_generation = mutations_ID[mutation_index].origin_generation;
			temp.origin_population = mutations_ID[mutation_index].origin_population;
			temp.origin_threadID = abs(mutations_ID[mutation_index].origin_threadID); //so the user doesn't get confused by the preservation flag on ID
			temp.reserved = mutations_ID[mutation_index].reserved;
			return temp;
		}
		fprintf(stderr,"mutation_ID error: requested mutation index out of bounds: mutation %d [0 %d)\n",mutation_index,maximal_num_mutations()); exit(1);
	}else{ fprintf(stderr,"mutation_ID error: empty allele_trajectories\n"); exit(1); }
}

inline void allele_trajectories::delete_time_sample(int sample_index){
	if(sample_index >= 0 && sample_index < num_samples){
		if(num_samples == 1){ reset(); }
		else{
			delete time_samples[sample_index];
			time_sample ** temp = new time_sample * [num_samples-1];
			for(int i = 0; i < num_samples; i++){
				if(i < sample_index){ temp[i] = time_samples[i]; }
				else if (i > sample_index){ temp[i-1] = time_samples[i]; }
			}
			delete [] time_samples;
			time_samples = temp;
			num_samples -= 1;
			all_mutations = time_samples[num_samples-1]->num_mutations; //new maximal number of mutations if last time sample has been deleted, moves the apparent length of mutID array, but does not delete extra data
		}
	}else{
		if(!time_samples || num_samples == 0){ fprintf(stderr,"delete_time_sample error: empty allele_trajectories\n"); exit(1); }
		fprintf(stderr,"delete_time_sample error: requested sample index out of bounds: sample %d [0 %d)\n",sample_index,num_samples); exit(1);
	}
}

inline void allele_trajectories::reset(){
	if(time_samples){
		for(int i = 0; i < num_samples; i++){ delete time_samples[i]; }
		delete [] time_samples;
	}
	if(mutations_ID){ delete [] mutations_ID; }
	time_samples = NULL; num_samples = 0; mutations_ID = NULL; all_mutations = 0;
	sim_run_constants = sim_constants();
	sim_input_constants = sim_constants();
}

inline allele_trajectories::~allele_trajectories(){ reset(); }

inline void allele_trajectories::initialize_sim_result_vector(int new_length){
	sim_constants temp = sim_input_constants; //store sim_input_constants as in this context they are still valid
	reset(); //overwrite old data if any
	num_samples = new_length;
	time_samples = new time_sample *[num_samples];
	for(int i = 0; i < num_samples; i++){ time_samples[i] = new time_sample(); }
	sim_input_constants = temp;
	sim_run_constants = sim_input_constants;
}

inline void swap(allele_trajectories & a, allele_trajectories & b){
	//can use for move constructor/assignment when moving to CUDA 7+ and C++11: http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
	allele_trajectories::sim_constants temp = a.sim_input_constants;
	a.sim_input_constants = b.sim_input_constants;
	b.sim_input_constants = temp;
	temp = a.sim_run_constants;
	a.sim_run_constants = b.sim_run_constants;
	b.sim_run_constants = temp;
	std::swap(a.all_mutations,b.all_mutations);
	std::swap(a.num_samples,b.num_samples);
	std::swap(a.mutations_ID,b.mutations_ID);
	std::swap(a.time_samples,b.time_samples);
}

} /* ----- end namespace GO_Fish ----- */

#endif /* INLINE_GOFISH_DATA_FUNCTIONS_HPP_ */
