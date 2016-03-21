#include "data_structs.h"

sim_result::sim_result(): num_populations(0), num_mutations(0), num_sites(0), total_generations(0) { mutations_freq = NULL; mutations_ID = NULL; extinct = NULL; }

void sim_result::store_sim_result(sim_result & out, sim_struct & mutations, int num_sites, int total_generations, cudaStream_t * control_streams, cudaEvent_t * control_events){
	out.num_populations = mutations.h_num_populations;
	out.num_mutations = mutations.h_mutations_Index;
	out.num_sites = num_sites;
	out.total_generations = total_generations;
	cudaCheckErrors(cudaMallocHost((void**)&out.mutations_freq,out.num_populations*out.num_mutations*sizeof(float)),total_generations,-1); //should allow for simultaneous transfer to host
	cudaCheckErrorsAsync(cudaMemcpy2DAsync(out.mutations_freq, out.num_mutations*sizeof(float), mutations.d_prev_freq, mutations.h_array_Length*sizeof(float), out.num_mutations*sizeof(float), out.num_populations, cudaMemcpyDeviceToHost, control_streams[1]),total_generations,-1); //removes padding
	cudaCheckErrors(cudaMallocHost((void**)&out.mutations_ID, out.num_mutations*sizeof(mutID)),total_generations,-1);
	cudaCheckErrorsAsync(cudaMemcpyAsync(out.mutations_ID, mutations.d_mutations_ID, out.num_mutations*sizeof(int4), cudaMemcpyDeviceToHost, control_streams[2]),total_generations,-1); //mutations array is 1D
	out.extinct = new bool[out.num_populations];
	for(int i = 0; i < out.num_populations; i++){ out.extinct[i] = mutations.h_extinct[i]; }


	cudaCheckErrorsAsync(cudaEventRecord(control_events[1],control_streams[1]),total_generations,-1);
	cudaCheckErrorsAsync(cudaEventRecord(control_events[2],control_streams[2]),total_generations,-1);
	cudaCheckErrorsAsync(cudaStreamWaitEvent(control_streams[0],control_events[1],0),total_generations,-1); //if compacting is about to happen, don't compact until results are compiled
	cudaCheckErrorsAsync(cudaStreamWaitEvent(control_streams[0],control_events[2],0),total_generations,-1);
	//1 round of migration_selection_drift and add_new_mutations can be done simultaneously with above as they change d_mutations_freq array, not d_prev_freq
}

sim_result::~sim_result(){ if(mutations_freq){ cudaFreeHost(mutations_freq); } if(mutations_ID){ cudaFreeHost(mutations_ID); } if(extinct){ delete [] extinct; } }


sim_struct::sim_struct(): h_num_populations(0), h_array_Length(0), h_mutations_Index(0), warp_size(0) { d_mutations_freq = NULL; d_prev_freq = NULL; d_mutations_ID = NULL; h_new_mutation_Indices = NULL; h_extinct = NULL;}
sim_struct::~sim_struct(){ cudaCheckErrorsAsync(cudaFree(d_mutations_freq),-1,-1); cudaCheckErrorsAsync(cudaFree(d_prev_freq),-1,-1); cudaCheckErrorsAsync(cudaFree(d_mutations_ID),-1,-1); if(h_new_mutation_Indices) { delete [] h_new_mutation_Indices; } if(h_extinct){ delete [] h_extinct; } }

sfs::sfs(): num_populations(0), num_sites(0), total_generations(0) {frequency_spectrum = NULL; frequency_age_spectrum = NULL; populations = NULL; num_samples = NULL;}
sfs::~sfs(){ if(frequency_spectrum){ delete[] frequency_spectrum; } if(frequency_age_spectrum){ delete[] frequency_age_spectrum; } if(populations){ delete[] populations; } if(num_samples){ delete[] num_samples; }}
