/*
 * spectrum.cu
 *
 *      Author: David Lawrie
 */

#include "spectrum.h"
#include "shared.cuh"

namespace SPECTRUM{

class transfer_allele_trajectories{

	struct time_sample{
		float * mutations_freq; //allele frequency of mutations in final generation
		GO_Fish::mutID * mutations_ID; //unique ID consisting of generation, population, threadID, and device
		bool * extinct; //extinct[pop] == true, flag if population is extinct by end of simulation
		int * Nchrom_e; //effective number of chromosomes in each population
		int num_populations; //number of populations in freq array (array length, rows)
		int num_mutations; //number of mutations in array (array length for age/freq, columns)
		float num_sites; //number of sites in simulation
		int sampled_generation; //number of generations in the simulation at point of sampling

		time_sample(): num_populations(0), num_mutations(0), num_sites(0), sampled_generation(0) { mutations_freq = 0; mutations_ID = 0; extinct = 0; Nchrom_e = 0; }
		time_sample(const GO_Fish::allele_trajectories & in, int sample_index): num_populations(in[sample_index]->num_populations), num_mutations(in[sample_index]->num_mutations), num_sites(in[sample_index]->num_sites), sampled_generation(in[sample_index]->sampled_generation){
			mutations_freq = in[sample_index]->mutations_freq;
			mutations_ID = in[sample_index]->mutations_ID;
			extinct = in[sample_index]->extinct;
			Nchrom_e = in[sample_index]->Nchrom_e;
		}
		~time_sample(){ mutations_freq = 0; mutations_ID = 0; extinct = 0; Nchrom_e = 0; } //don't actually delete information, just null pointers as this just points to the real data held
	};

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
		sim_input_constants(const GO_Fish::allele_trajectories & in){
			seed1 = in.sim_input_constants.seed1;
			seed2 = in.sim_input_constants.seed2;
			num_generations = in.sim_input_constants.num_generations;
			num_sites = in.sim_input_constants.num_sites;
			num_discrete_DFE_categories = in.sim_input_constants.num_discrete_DFE_categories;
			num_populations = in.sim_input_constants.num_populations;
			init_mse = in.sim_input_constants.init_mse;
			prev_sim_sample = in.sim_input_constants.prev_sim_sample;
			compact_rate = in.sim_input_constants.compact_rate;
			device = in.sim_input_constants.device;
		}
	}sim_input_constants;
	//----- end -----

public:

	transfer_allele_trajectories(): length(0) { time_samples = 0; }

	transfer_allele_trajectories(const GO_Fish::allele_trajectories & in): sim_input_constants(in){
		if(!in.time_samples || in.length == 0){ fprintf(stderr,"error transferring allele_trajectories to spectrum: empty allele_trajectories\n"); exit(1); }
		length = in.length;
		time_samples = new time_sample *[length];

		for(int i = 0; i < length; i++){ time_samples[i] = new time_sample(in,i); }
	}

	friend sfs site_frequency_spectrum(const GO_Fish::allele_trajectories & all_results, int sample_index, int population_index, int cuda_device);

	~transfer_allele_trajectories(){ time_samples = 0; length = 0; } //don't actually delete anything, this is just a pointer class, actual data held by GO_Fish::trajectory
};

__global__ void simple_hist(int * out_histogram, float * in_mutation_freq, int num_samples, int num_mutations, int num_sites){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;

	for(int id = myID; id < num_mutations; id+= blockDim.x*gridDim.x){
		int index = round(num_samples*in_mutation_freq[id]);
		atomicAdd(&out_histogram[index],1);
	}
	if(myID == 0){  out_histogram[0] = num_sites - num_mutations;  }
}

sfs::sfs(): num_populations(0), num_sites(0), sampled_generation(0) {frequency_spectrum = NULL; populations = NULL; num_samples = NULL;}
sfs::~sfs(){ if(frequency_spectrum){ cudaCheckErrors(cudaFreeHost(frequency_spectrum),-1,-1); } if(populations){ delete[] populations; } if(num_samples){ delete[] num_samples; }}

//single-population sfs
sfs site_frequency_spectrum(const GO_Fish::allele_trajectories & all_results, int sample_index, int population_index, int cuda_device){

	set_cuda_device(cuda_device);

	cudaStream_t stream;

	cudaCheckErrors(cudaStreamCreate(&stream),-1,-1);

	float * d_mutations_freq;
	int * d_histogram, * h_histogram;
	transfer_allele_trajectories sample(all_results);
	if(!(sample_index >= 0 && sample_index < sample.length)){ fprintf(stderr,"site_frequency_spectrum error: requested sample index out of bounds: sample %d\t[0\t %d)\n",sample_index,sample.length); exit(1); }
	if(!(population_index >= 0 && population_index < sample.sim_input_constants.num_populations)){ fprintf(stderr,"site_frequency_spectrum error: requested population index out of bounds: population %d\t[0\t %d)\n",population_index,sample.sim_input_constants.num_populations); exit(1); }
	int num_levels = sample.time_samples[sample_index]->Nchrom_e[population_index]+1;

	cudaCheckErrorsAsync(cudaMalloc((void**)&d_mutations_freq, sample.time_samples[sample_index]->num_mutations*sizeof(float)),-1,-1);
	cudaCheckErrorsAsync(cudaMalloc((void**)&d_histogram, num_levels*sizeof(int)),-1,-1);
	cudaCheckErrorsAsync(cudaMemsetAsync(d_histogram, 0, num_levels*sizeof(int), stream),-1,-1);
	cudaCheckErrorsAsync(cudaMemcpyAsync(d_mutations_freq, &sample.time_samples[sample_index]->mutations_freq[population_index*sample.time_samples[sample_index]->num_mutations], sample.time_samples[sample_index]->num_mutations*sizeof(float), cudaMemcpyHostToDevice, stream),-1,-1);

	simple_hist<<<50,1024,0,stream>>>(d_histogram, d_mutations_freq, sample.time_samples[sample_index]->Nchrom_e[population_index], sample.time_samples[sample_index]->num_mutations, sample.time_samples[sample_index]->num_sites);
	cudaCheckErrorsAsync(cudaPeekAtLastError(),-1,-1);

	cudaCheckErrors(cudaMallocHost((void**)&h_histogram, num_levels*sizeof(int)),-1,-1);
	cudaCheckErrorsAsync(cudaMemcpyAsync(h_histogram, d_histogram, num_levels*sizeof(int), cudaMemcpyDeviceToHost, stream),-1,-1);

	if(cudaStreamQuery(stream) != cudaSuccess){ cudaCheckErrors(cudaStreamSynchronize(stream), -1, -1); } //wait for writes to host to finish

	sfs mySFS;
	mySFS.frequency_spectrum = h_histogram;
	mySFS.num_populations = 1;
	mySFS.num_samples = new int[1];
	mySFS.num_samples[0] = num_levels;
	mySFS.num_sites = sample.time_samples[sample_index]->num_sites;
	mySFS.populations = new int[1];
	mySFS.populations[0] = population_index;
	mySFS.sampled_generation = sample.time_samples[sample_index]->sampled_generation;

	//cudaCheckErrorsAsync(cudaFree(d_temp_storage),-1,-1);
	cudaCheckErrorsAsync(cudaFree(d_mutations_freq),-1,-1);
	cudaCheckErrorsAsync(cudaFree(d_histogram),-1,-1);
	cudaCheckErrorsAsync(cudaStreamDestroy(stream),-1,-1)

	return mySFS;
}

} /*----- end namespace SPECTRUM ----- */
