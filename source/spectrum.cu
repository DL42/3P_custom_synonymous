/*
 * spectrum.cu
 *
 *      Author: David Lawrie
 */

#include "../include/spectrum.h"
#include "../source/shared.cuh"
#include <cub/device/device_scan.cuh>

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
		time_sample(const GO_Fish::allele_trajectories & in, int sample_index): num_populations(in.time_samples[sample_index]->num_populations), num_mutations(in.time_samples[sample_index]->num_mutations), num_sites(in.time_samples[sample_index]->num_sites), sampled_generation(in.time_samples[sample_index]->sampled_generation){
			mutations_freq = in.time_samples[sample_index]->mutations_freq;
			mutations_ID = in.time_samples[sample_index]->mutations_ID;
			extinct = in.time_samples[sample_index]->extinct;
			Nchrom_e = in.time_samples[sample_index]->Nchrom_e;
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

	friend sfs site_frequency_spectrum(const GO_Fish::allele_trajectories & all_results, const int sample_index, const int population_index, const unsigned int sample_size, int cuda_device);

	~transfer_allele_trajectories(){ time_samples = 0; length = 0; } //don't actually delete anything, this is just a pointer class, actual data held by GO_Fish::trajectory
};

__global__ void population_hist(unsigned int * out_histogram, float * in_mutation_freq, int Nchrome_e, int num_mutations, int num_sites){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;

	for(int id = myID; id < num_mutations; id+= blockDim.x*gridDim.x){
		int index = round(Nchrome_e*in_mutation_freq[id]);
		atomicAdd(&out_histogram[index],1);
	}
	if(myID == 0){  out_histogram[0] = num_sites - num_mutations;  }
}

__global__ void uint_to_float(float * out_array, unsigned int * in_array, int N){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < N; id+= blockDim.x*gridDim.x){ out_array[id] = in_array[id]; }
}

/*    function [P_samp] = sample_prob(Npop,Nsamp)
        x = (0:(2*Npop-1))/(2*Npop);
        P_samp = zeros((Nsamp+1),length(x));
        for k = 0:Nsamp
            P_samp((k+1),:) = binopdf(k,Nsamp,x);
        end
    end

    function [prob] = prob(k,rho,P_samp)
        if(length(P_samp) > length(rho))
            prob = dot(rho,P_samp((k+1),(2:length(P_samp))));
        else
            prob = dot(rho,P_samp((k+1),:));
        end

    end*/

//pdivq = p/q
__global__ void binom_fract(float * binom_fract, float q, float pdivq, int half_n, int n){
	int myIDx =  blockIdx.x*blockDim.x + threadIdx.x;

	for(int idx = (myIDx+1); idx < half_n; idx+= blockDim.x*gridDim.x){ binom_fract[idx] =  ((n+1.f-idx)/((float)idx))*pdivq; }
	if(myIDx == 0){ binom_fract[0] = pow(q,n); }
}

struct CustomMultiply
{
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return a*b; }
};

__global__ void print_Device_array_float(float * array, int num){

		//if(i%1000 == 0){ printf("\n"); }
	for(int j = 0; j < num; j++){ printf("%d: %f\t",j,array[j]); }
	printf("\n");
}


sfs::sfs(): num_populations(0), num_sites(0), sampled_generation(0) {frequency_spectrum = NULL; populations = NULL; sample_size = NULL;}
sfs::~sfs(){ if(frequency_spectrum){ cudaCheckErrors(cudaFreeHost(frequency_spectrum),-1,-1); frequency_spectrum = NULL; } if(populations){ delete[] populations; populations = NULL; } if(sample_size){ delete[] sample_size; sample_size = NULL; }}

//single-population sfs
sfs site_frequency_spectrum(const GO_Fish::allele_trajectories & all_results, const int sample_index, const int population_index, const unsigned int sample_size, int cuda_device){

	set_cuda_device(cuda_device);

	cudaStream_t stream;

	cudaCheckErrors(cudaStreamCreate(&stream),-1,-1);

	float * d_mutations_freq;
	unsigned int * d_pop_histogram;
	float * d_histogram, * h_histogram;
	transfer_allele_trajectories sample(all_results);
	if(!(sample_index >= 0 && sample_index < sample.length) || !(population_index >= 0 && population_index < sample.sim_input_constants.num_populations)){
		fprintf(stderr,"site_frequency_spectrum error: requested indices out of bounds: sample %d\t[0 %d)\tpopulation %d\t[0 %d)\n",sample_index,sample.length,population_index,sample.sim_input_constants.num_populations); exit(1);
	}

	int num_levels = sample_size;
	int population_size = sample.time_samples[sample_index]->Nchrom_e[population_index];
	if(sample_size == 0){ num_levels = population_size; }

	cudaCheckErrorsAsync(cudaMalloc((void**)&d_mutations_freq, sample.time_samples[sample_index]->num_mutations*sizeof(float)),-1,-1);
	cudaCheckErrorsAsync(cudaMalloc((void**)&d_pop_histogram, num_levels*sizeof(unsigned int)),-1,-1);
	cudaCheckErrorsAsync(cudaMalloc((void**)&d_histogram, num_levels*sizeof(float)),-1,-1);
	cudaCheckErrorsAsync(cudaMemsetAsync(d_pop_histogram, 0, num_levels*sizeof(unsigned int), stream),-1,-1);
	cudaCheckErrorsAsync(cudaMemcpyAsync(d_mutations_freq, &sample.time_samples[sample_index]->mutations_freq[population_index*sample.time_samples[sample_index]->num_mutations], sample.time_samples[sample_index]->num_mutations*sizeof(float), cudaMemcpyHostToDevice, stream),-1,-1);

	population_hist<<<50,1024,0,stream>>>(d_pop_histogram, d_mutations_freq, population_size, sample.time_samples[sample_index]->num_mutations, sample.time_samples[sample_index]->num_sites);
	cudaCheckErrorsAsync(cudaPeekAtLastError(),-1,-1);
	/*         for k = 1:(N_samp-1)
	            mix_prob = prob(k,rho_mix,P_samp);
	            %G(k) = round(mix_prob*m_mix);
	            G(k) = mix_prob*m_mix;
	            m_samp = m_samp + G(k);
	        end   */

	int num_threads = 1024;
	int num_blocks = max(num_levels/num_threads,1);
	if(sample_size == 0){
		uint_to_float<<<num_blocks,num_threads,0,stream>>>(d_histogram, d_pop_histogram, num_levels);
		cudaCheckErrorsAsync(cudaPeekAtLastError(),-1,-1);
	}
	//else{
		int temp = num_levels;
		num_levels = 1000;
		num_threads = 128;
		num_blocks = min(num_levels/num_threads,1);
		int half_n;
		if(num_levels % 2 == 0){ half_n = num_levels/2+1; }
		else{ half_n = (num_levels+1)/2; }

		float * d_binom_fract;
		cudaCheckErrorsAsync(cudaMalloc((void**)&d_binom_fract, half_n*sizeof(float)),-1,-1);
		const dim3 gridsize(10,10,1);
		float p = (20000.f)/(float(population_size));
		float q = (population_size - 20000.f)/(float(population_size));
		p = 0.01;
		q = 0.99;
		//printf("p: ");
		float pdivq = (double)p/(double)q;
		binom_fract<<<50,128,0,stream>>>(d_binom_fract, q, pdivq, half_n, num_levels);
		cudaCheckErrorsAsync(cudaPeekAtLastError(),-1,-1);

		print_Device_array_float<<<1,1,0,stream>>>(d_binom_fract, 50);

		float * d_binom;
		CustomMultiply mul_op;
		cudaCheckErrorsAsync(cudaMalloc((void**)&d_binom, half_n*sizeof(float)),-1,-1);

		void *d_temp_storage = NULL;
		size_t temp_storage_bytes = 0;
		cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_binom_fract, d_binom, mul_op, half_n);
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_binom_fract, d_binom, mul_op, half_n);
		cudaCheckErrorsAsync(cudaFree(d_temp_storage),-1,-1);
		cudaCheckErrorsAsync(cudaFree(d_binom_fract),-1,-1);

		print_Device_array_float<<<1,1,0,stream>>>(d_binom, 50);

		cudaCheckErrorsAsync(cudaFree(d_binom),-1,-1);
		num_levels = temp;
	//}

	cudaCheckErrors(cudaMallocHost((void**)&h_histogram, num_levels*sizeof(float)),-1,-1);
	cudaCheckErrorsAsync(cudaMemcpyAsync(h_histogram, d_histogram, num_levels*sizeof(int), cudaMemcpyDeviceToHost, stream),-1,-1);

	if(cudaStreamQuery(stream) != cudaSuccess){ cudaCheckErrors(cudaStreamSynchronize(stream), -1, -1); } //wait for writes to host to finish

	sfs mySFS;
	mySFS.frequency_spectrum = h_histogram;
	mySFS.num_populations = 1;
	mySFS.sample_size = new int[1];
	mySFS.sample_size[0] = num_levels;
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
