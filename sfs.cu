#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <limits>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <helper_math.h>
#include <Random123/philox.h>
#include <Random123/features/compilerfeatures.h>
#define CUB_STDERR
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>

using namespace std;
using namespace cub;
using namespace r123;

// uint_float_01: Input is a W-bit integer (unsigned).  It is multiplied
// by Float(2^-W) and added to Float(2^(-W-1)).  A good compiler should
// optimize it down to an int-to-float conversion followed by a multiply
// and an add, which might be fused, depending on the architecture.
//
// If the input is a uniformly distributed integer, then the
// result is a uniformly distributed floating point number in [0, 1].
// The result is never exactly 0.0.
// The smallest value returned is 2^-W.
// Let M be the number of mantissa bits in Float.
// If W>M  then the largest value retured is 1.0.
// If W<=M then the largest value returned is the largest Float less than 1.0.
__host__ __device__ float uint_float_01(unsigned int in){
	//(mostly) stolen from Philox code "uniform.hpp"
	R123_CONSTEXPR float factor = float(1.)/(UINT_MAX + float(1.));
	R123_CONSTEXPR float halffactor = float(0.5)*factor;
    return in*factor + halffactor;
}

__host__ __device__ int4 round(float4 f){ return make_int4(round(f.x), round(f.y), round(f.z), round(f.w)); }

__host__ __device__ float4 exp(float4 f){ return make_float4(exp(f.x),exp(f.y),exp(f.z),exp(f.w)); }

__host__ __device__ double4 expd(float4 d){ return make_double4(exp(double(d.x)),exp(double(d.y)),exp(double(d.z)),exp(double(d.w))); }

inline __host__ __device__ double4 operator*(int a, double4 b){ return make_double4(a * b.x, a * b.y, a * b.z, a * b.w); }

inline __host__ __device__ double4 operator*(double a, float4 b){ return make_double4(a * b.x, a * b.y, a * b.z, a * b.w); }

inline __host__ __device__ double4 operator*(double4 a, double4 b){ return make_double4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }

inline __host__ __device__ double4 operator+(double4 b, double a){ return make_double4(a + b.x, a + b.y, a + b.z, a + b.w); }

inline __host__ __device__ float4 operator/(double4 a, double4 b){ return make_float4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w); }

inline __host__ __device__ float4 operator-(float a, float4 b){ return make_float4((a-b.x), (a-b.y), (a-b.z), (a-b.w)); }

__host__ __device__  uint4 Philox(int k, int step, int seed, int population, int round){
	typedef Philox4x32_R<8> P; //can change the 10 rounds of bijection down to 8 (lowest safe limit) to get possible extra speed!
	P rng;

	P::key_type key = {{0xbeeff00d, seed}}; //random int to set key space + seed
	P::ctr_type count = {{k, step, population, round}};

	union {
		P::ctr_type c;
		uint4 i;
	}u;

	u.c = rng(count, key);

	return u.i;
}



__host__ __device__ int poiscdfinv(float p, float mean){
	float e = exp(-1 * mean);
	float lambda_j = 1;
	float factorial = 1;
	int j = 0;

	float sum = lambda_j/factorial;
	float cdf = e*sum;
	if(cdf >= p){ return j; }

	j = 1;
	lambda_j = mean;
	sum += lambda_j/factorial;
	cdf = e*sum;
	if(cdf >= p){ return j; }
	float end = mean + 7*sqrtf(mean);
	j = 2;
	for(j = 2; j < end; j++){ //stops after the cdf surpasses p or j exceeds 7*standard deviation+mean (testing reveals rarely gets there anyway when putting in 1 for p)
		lambda_j *= mean;
		factorial*= j;
		sum += lambda_j/factorial;
		cdf = e*sum;
		if(cdf >= p){ return j; }
	}

	return j;
}

__host__ __device__ int RandBinom(float p, float N, int k, int step, int seed, int population, int start_round){
//only for use when N is small
	int j = 0;
	int counter = 0;
	
	union {
		uint h[4];
		uint4 i;
	}u;
	
	while(counter < N){
		u.i = Philox(k, step, seed, population, start_round+counter);
		
		int counter2 = 0;
		while(counter < N && counter2 < 4){
			if(uint_float_01(u.h[counter2]) <= p){ j++; }
			counter++;
			counter2++;
		}
	}
	
	return j;
}

__host__ __device__ int Rand1(float mean, float var, float p, float N, int k, int step, int seed, int population){

	if(N <= 50){ return RandBinom(p, N, k, step, seed, population, 0); }
	uint4 i = Philox(k, step, seed, population, 0);
	if(mean <= 10){ return poiscdfinv(uint_float_01(i.x), mean); }
	else if(mean >= N-10){ return N - poiscdfinv(uint_float_01(i.x), N-mean); } //flip side of binomial, when 1-p is small
	return round(normcdfinv(uint_float_01(i.x))*sqrtf(var)+mean);
}

__device__ int Rand1(unsigned int i, float mean, float var, float N){
	if(mean <= 10){ return poiscdfinv(uint_float_01(i), mean); }
	else if(mean >= N-10){ return N - poiscdfinv(uint_float_01(i), N-mean); } //flip side of binomial, when 1-p is small
	return round(normcdfinv(uint_float_01(i))*sqrtf(var)+mean);
}

__device__ int4 Rand4(float4 mean, float4 var, float4 p, float N, int k, int step, int seed, int population){
	if(N <= 50){ return make_int4(RandBinom(p.x, N, k, step, seed, population, 0),RandBinom(p.y, N, k, step, seed, population, N),RandBinom(p.z, N, k, step, seed, population, 2*N),RandBinom(p.w, N, k, step, seed, population, 3*N)); }
	uint4 i = Philox(k, step, seed, population, 0);
	return make_int4(Rand1(i.x,mean.x, var.x, N), Rand1(i.y,mean.y, var.y, N), Rand1(i.z,mean.z, var.z, N), Rand1(i.w,mean.w, var.w, N));
}

template <typename Functor_selection>
__global__ void initialize_mse_frequency_array(int * freq_index, const int offset, const float mu, const int N, const float L, const Functor_selection sel_coeff, const float h, const int seed, const int population){
	//determines number of mutations at each frequency in the initial population, sets it equal to mutation-selection balance
	int myID = blockIdx.x*blockDim.x + threadIdx.x;

	for(int id = myID; id < (N-1)/4; id += blockDim.x*gridDim.x){ //exclusive, number of freq in pop is chromosome population size N-1
		float4 i = make_float4((4*id + 1),(4*id + 2),(4*id + 3),(4*id + 4))/N;
		float s = sel_coeff(population, 0, 0.5); //the equations below don't work for frequency-dependent selection anyway
		float4 lambda;
		if(s == 0){ lambda = 2*mu*L/i; }
		else{ lambda =  2*mu*L*((-1.0*expd(-1*(2*N*s)*(-1.f*i+1.0))+1.0)/((-1*exp(-1*(2*N*double(s)))+1)*i*(-1.0*i+1.0))); }
		reinterpret_cast<int4*>(freq_index)[offset/4 + id] = max(Rand4(lambda, lambda, make_float4(mu), L*N, 0, id, seed, population),make_int4(0)); //round(lambda);//// ////mutations are poisson distributed in each frequency class
		//printf(" %d %d %d %f %f %f %f %f %f %f %f \r", myID, id, population, i.x, i.y, i.z, i.w, lambda.x, lambda.y, lambda.z, lambda.w);
	}

	int next_offset = (int)(ceil((N-1)/4.f)*4);
	int id = myID + (N-1)/4*4; //all integers //right now only works if minimum of 3 threads are launched
	if(id < (N-1)){
		float i = (id+1.f)/N;
		float s = sel_coeff(population, 0, 0.5);
		float lambda;
		if(s == 0){ lambda = 2*mu*L/i; }
		else{ lambda =  2*mu*L*(1-exp(-1*(2*N*double(s))*(1-i)))/((1-exp(-1*(2*N*double(s))))*i*(1-i)); }
		freq_index[offset+id] = max(Rand1(lambda, lambda, mu, L*N, 0, id, seed, population),0);//round(lambda);// //  //mutations are poisson distributed in each frequency class
		//printf(" %d %d %d %f %f\r", myID, id, population, i, lambda);
	}else if(id >= (N-1) && id < next_offset){ freq_index[offset+id] = 0; } //ensures padding at end of population is set to 0
}

__global__ void initialize_mse_mutation_array(float * mutations, const int * freq_index, const int * scan_index, const int offset, const int N, const int population, const int num_populations, const int array_Length){
	//fills in mutation array using the freq and scan indices
	//y threads correspond to freq_index/scan_index indices, use grid-stride loops
	//x threads correspond to mutation array indices, use grid-stride loops
	//using scan number to define start of array, freq_index to define num_new_mutations_index (if 0 simply ignore) and myIDx used to calculate allele_count
	int myIDy = blockIdx.y*blockDim.y + threadIdx.y;
	for(int idy = myIDy; idy < (N-1); idy+= blockDim.y*gridDim.y){
		int myIDx = blockIdx.x*blockDim.x + threadIdx.x;
		int start = scan_index[offset+idy];
		int num_mutations = freq_index[offset+idy];
		float freq = (idy+1.f)/N;
		for(int idx = myIDx; idx < num_mutations; idx+= blockDim.x*gridDim.x){
			for(int pop = 0; pop < num_populations; pop++){ mutations[pop*array_Length + start + idx] = 0; }
			mutations[population*array_Length + start + idx] = freq;
		}
	}
}

/*__global__ void print_Device_array_int(int * array, int num){

	for(int i = 0; i < num; i++){
		if(i%1000 == 0){ printf("\r"); }
		printf("%d ",array[i]);
	}
}*/

/*__global__ void sum_Device_array_int(int * array, int num){
	int j = 0;
	for(int i = 0; i < num; i++){
		j += array[i];
	}
	printf("\r%d\r",j);
}*/

//calculates new frequencies for every mutation in the population
//seed for random number generator philox's key space, id, generation for its counter space in the pseudorandom sequence
template <typename Functor_migration, typename Functor_selection>
__global__ void migration_selection_drift(float * mutations_freq, const int mutations_Index, const int array_Length, const int N, const Functor_migration mig_prop, const Functor_selection sel_coeff, const float h, const float F, const int seed, const int population, const int num_populations, const int generation){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;

	for(int id = myID; id < mutations_Index/4; id+= blockDim.x*gridDim.x){
		float4 i_mig = make_float4(0);
		for(int pop = 0; pop < num_populations; pop++){
			float4 i = reinterpret_cast<float4*>(mutations_freq)[pop*array_Length/4+id]; //allele frequency in previous population //make sure array length is divisible by 4 (preferably divisible by 32 or warp_size)!!!!!!
			i_mig += mig_prop(pop,population,generation)*i;
		}
		float4 s = make_float4(sel_coeff(population,generation,i_mig.x),sel_coeff(population,generation,i_mig.y),sel_coeff(population,generation,i_mig.z),sel_coeff(population,generation,i_mig.w));
		float4 i_mig_sel = (s*i_mig*i_mig+i_mig+(F+h-h*F)*s*i_mig*(1-i_mig))/(i_mig*i_mig*s+(F+2*h-2*h*F)*s*i_mig*(1-i_mig)+1);
		float4 mean = i_mig_sel*N; //expected allele count in new generation
		int4 j_mig_sel_drift = clamp(Rand4(mean,(-1.f*i_mig_sel + 1.0)*mean,i_mig_sel,N,(id + 2),generation,seed,population), 0, N);
		reinterpret_cast<float4*>(mutations_freq)[population*array_Length/4+id] = make_float4(j_mig_sel_drift)/N; //final allele freq in new generation //make sure array length is divisible by 4 (preferably 32/warp_size)!!!!!!
	}
	int id = myID + mutations_Index/4 * 4;  //right now only works if minimum of 3 threads are launched
	if(id < mutations_Index){
		float i_mig = 0;
		for(int pop = 0; pop < num_populations; pop++){
			float i = mutations_freq[pop*array_Length+id]; //allele frequency in previous population
			i_mig += mig_prop(pop,population,generation)*i;
		}
		float s = sel_coeff(0,generation,i_mig);
		float i_mig_sel = (s*i_mig*i_mig+i_mig+(F+h-h*F)*s*i_mig*(1-i_mig))/(i_mig*i_mig*s+(F+2*h-2*h*F)*s*i_mig*(1-i_mig)+1);
		float mean = i_mig_sel*N; //expected allele count in new generation
		int j_mig_sel_drift = clamp(Rand1(mean,(1.0-i_mig_sel)*mean,i_mig_sel,N,(id + 2),generation,seed,population), 0, N);
		mutations_freq[population*array_Length+id] = float(j_mig_sel_drift)/N; //final allele freq in new generation
	}
}

__global__ void add_new_mutations(float * mutations_freq, int * mutations_age, const int prev_mutations_Index, const int new_mutations_Index, const int array_Length, float freq, const int population, const int num_populations, const int generation){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; (id < (new_mutations_Index-prev_mutations_Index)) && ((id + prev_mutations_Index) < array_Length); id+= blockDim.x*gridDim.x){
		for(int pop = 0; pop < num_populations; pop++){ mutations_freq[(pop*array_Length+prev_mutations_Index+id)] = 0; }
		mutations_freq[(population*array_Length+prev_mutations_Index+id)] = freq;
		mutations_age[(prev_mutations_Index+id)] = generation;
	}
}

__device__ int4 boundary(float4 freq){
	return make_int4((freq.x <= 0.f || freq.x >= 1.f), (freq.y <= 0.f || freq.y >= 1.f), (freq.z <= 0.f || freq.z >= 1.f), (freq.w <= 0.f || freq.w >= 1.f));
}

__device__ int boundary(float freq){
	return (freq <= 0.f || freq >= 1.f);
}

__global__ void flag_segregating_mutations(int * flag, const float * const mutations_freq, const int num_populations, const int mutations_Index, const int array_Length){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < mutations_Index/4; id+= blockDim.x*gridDim.x){
		int4 i = make_int4(1);
		for(int pop = 0; pop < num_populations; pop++){ i *= boundary(reinterpret_cast<const float4*>(mutations_freq)[pop*array_Length/4+id]); } //make sure array length is divisible by 4 (preferably 32/warp_size)!!!!!!
		reinterpret_cast<int4*>(flag)[id] = make_int4(!i.x,!i.y,!i.z,!i.w); //1 if allele is segregating in any population, 0 otherwise
	}
	int id = myID + mutations_Index/4 * 4;  //right now only works if minimum of 3 threads are launched
	if(id < mutations_Index){
		int i = 1;
		for(int pop = 0; pop < num_populations; pop++){ i *= boundary(mutations_freq[pop*array_Length+id]); }
		flag[id] = !i; //1 if allele is segregating in any population, 0 otherwise
	}
}

__global__ void scatter_arrays(float * new_mutations_freq, int * new_mutations_age, const float * const mutations_freq, const int * const mutations_age, const int * const flag, const int * const scan_Index, const int mutations_Index, const int new_array_Length, const int old_array_Length){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	int population = blockIdx.y;
	for(int id = myID; id < mutations_Index; id+= blockDim.x*gridDim.x){
		if(flag[id]){
			int index = scan_Index[id];
			new_mutations_freq[population*new_array_Length+index] = mutations_freq[population*old_array_Length+id];
			if(population == 0){ new_mutations_age[index] = mutations_age[id]; }
		}
	}
}

__global__ void refactor_mutation_age(int * mutation_age, int mutations_Index, int total_generations){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < mutations_Index/4; id+= blockDim.x*gridDim.x){
		reinterpret_cast<float4*>(mutation_age)[id] =  reinterpret_cast<const float4*>(mutation_age)[id] - total_generations;
	}
	int id = myID + mutations_Index/4 * 4;  //right now only works if minimum of 3 threads are launched
	if(id < mutations_Index){ mutation_age[id] = mutation_age[id] - total_generations; }
}

struct mig_prop
{
	float m;
	mig_prop(float m) : m(m){ }
	__device__ __forceinline__ float operator()(const int pop_FROM, const int pop_TO, const int generation) const{
		if(pop_FROM == pop_TO){ return 1-m; }
		return m;
	}
};


struct sel_coeff
{
	float s;
	sel_coeff(float s) : s(s){ }
	__device__ __forceinline__ float operator()(const int population, const int generation, const float freq) const{
		return s;
	}
};

struct demography
{
	int N;
	int total_generations;
	demography(int N, int total_generations) : N(N), total_generations(total_generations){ }
	__host__ __forceinline__ int operator()(const int population, const int generation) const{
		if(generation < total_generations){ return N; }
		return -1;
	}
};

struct inbreeding
{
	float F;
	inbreeding(float F) : F(F){ }
	__host__ __forceinline__ float operator()(const int population, const int generation) const{
		return F;
	}
};

struct mutation
{
	float mu;
	mutation(float mu) : mu(mu){ }
	__host__ __forceinline__ float operator()(const int population, const int generation) const{
		return mu;
	}
};

//for internal function passing
struct sim_struct{
	//device arrays
	float * d_mutations_freq; //allele frequency of current mutations
	int * d_mutations_age;  //generation when mutation entered the population

	int h_num_populations; //number of populations (# rows for freq)
	int h_array_Length; //full length of the mutation array, total number of mutations across all populations (# columns for freq)
	int h_mutations_Index; //number of mutations in the population (last mutation is at h_mutations_Index-1)
	int * h_new_mutation_Indices; //indices of new mutations, current age/freq index of mutations is at position 0, index for mutations in population 0 to be added to array is at position 1, etc ...

	sim_struct(): h_num_populations(0), h_array_Length(0), h_mutations_Index(0) { d_mutations_freq = NULL; d_mutations_age = NULL; h_new_mutation_Indices = NULL;}
	~sim_struct(){ cudaFree(d_mutations_freq); cudaFree(d_mutations_age); delete h_new_mutation_Indices; }
};

//for final result output
struct sim_result{
	float * mutations_freq; //allele frequency of mutations in final generation
	int * mutations_age; //allele age of mutations in final generation (0 most recent, negative values for older mutations)
	int num_populations; //number of populations in freq array (array length, rows)
	int num_mutations; //number of mutations in array (array length for age/freq, columns)
	int num_sites; //number of sites in simulation
	int total_generations; //number of generations in the simulation

	sim_result() : num_populations(0), num_mutations(0), num_sites(0), total_generations(0) { mutations_freq = NULL; mutations_age = NULL; }
	sim_result(sim_struct & mutations, int num_sites, int total_generations) : num_populations(mutations.h_num_populations), num_mutations(mutations.h_mutations_Index), num_sites(num_sites), total_generations(total_generations){
		mutations_freq = new float[num_populations*num_mutations];
		cudaMemcpyAsync(mutations_freq, mutations.d_mutations_freq, num_populations*num_mutations*sizeof(float), cudaMemcpyDeviceToHost);
		mutations_age = new int[num_mutations];
		cudaMemcpy(mutations_age, mutations.d_mutations_age, num_mutations*sizeof(int), cudaMemcpyDeviceToHost);
	}
	~sim_result(){ if(mutations_freq){ delete mutations_freq; } if(mutations_age){ delete mutations_age; } }
};

template <typename Functor_mu, typename Functor_dem>
__host__ __forceinline__ void set_Index_Length(sim_struct & mutations, const int num_mutations, const Functor_mu mu_rate, const Functor_dem demography, const float num_sites, const int compact_rate, const int num_populations, const int generation){
	mutations.h_mutations_Index = num_mutations;
	mutations.h_array_Length = mutations.h_mutations_Index;
	for(int gen = generation; gen < (generation+compact_rate); gen++){
		if(demography(0,gen) == -1){ break; } //reference population has ended
		for(int pop = 0; pop < num_populations; pop++){  mutations.h_array_Length += mu_rate(pop,gen)*demography(pop,gen)*num_sites + 7*sqrtf(mu_rate(pop,gen)*demography(pop,gen)*num_sites); }
	}
	mutations.h_array_Length = (int)(ceil(mutations.h_array_Length/32.f)*32); //replace with variable for warp size, coalesces memory access for multiple populations
}

template <typename Functor_mutation, typename Functor_demography>
__host__ __forceinline__ void calc_new_mutations_Index(sim_struct & mutations, const Functor_mutation mu_rate, const Functor_demography demography, const float L, const int seed, const int num_populations, const int generation){
	int num_new_mutations = 0;
	mutations.h_new_mutation_Indices[0] = mutations.h_mutations_Index;
	for(int pop = 0; pop < num_populations; pop++){
		int N = demography(pop, generation);
		float mu = mu_rate(pop, generation);
		float lambda = mu*N*L;
		int temp = max(Rand1(lambda, lambda, mu, N*L, 1, generation, seed, pop),0);
		num_new_mutations += temp;
		mutations.h_new_mutation_Indices[pop+1] = num_new_mutations + mutations.h_mutations_Index;
	}
}

template <typename Functor_mutation, typename Functor_demography, typename Functor_selection>
__host__ __forceinline__ void initialize_mse(sim_struct & mutations, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_selection s, const int h, const float num_sites, const int seed, const int compact_rate){

	int Nfreq = 0; //number of frequencies
	for(int pop = 0; pop < mutations.h_num_populations; pop++){ Nfreq += (int)(ceil((demography(pop,0) - 1)/4.f)*4); } //adds a little padding to ensure distances between populations in array are a multiple of 4

	int * d_freq_index;
	cudaMalloc((void**)&d_freq_index, Nfreq*sizeof(int));
	int * d_scan_index;
	cudaMalloc((void**)&d_scan_index,Nfreq*sizeof(int));

	int offset = 0;
	for(int pop = 0; pop < mutations.h_num_populations; pop++){
		initialize_mse_frequency_array<<<6,1024>>>(d_freq_index, offset, mu_rate(pop,0), demography(pop,0), num_sites, s, h, seed, pop);
		offset += (int)(ceil((demography(pop,0) - 1)/4.f)*4);
	}

	void * d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_freq_index, d_scan_index, Nfreq);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_freq_index, d_scan_index, Nfreq);
	cudaFree(d_temp_storage);

	int prefix_sum_result;
	int final_freq_count;
	//final index is Nfreq-1
	cudaMemcpy(&prefix_sum_result, &d_scan_index[(Nfreq-1)], sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&final_freq_count, &d_freq_index[(Nfreq-1)], sizeof(int), cudaMemcpyDeviceToHost);
	int num_mutations = prefix_sum_result+final_freq_count;
	set_Index_Length(mutations, num_mutations, mu_rate, demography, num_sites, compact_rate, mutations.h_num_populations, 0);
	cout<<"initial length " << mutations.h_array_Length << endl;

	cudaMalloc((void**)&mutations.d_mutations_freq, mutations.h_num_populations*mutations.h_array_Length*sizeof(float));
	cudaMalloc((void**)&mutations.d_mutations_age, mutations.h_num_populations*mutations.h_array_Length*sizeof(int));

	const dim3 blocksize(4,256,1);
	const dim3 gridsize(16,32,1);
	offset = 0;
	for(int pop = 0; pop < mutations.h_num_populations; pop++){
		initialize_mse_mutation_array<<<gridsize,blocksize>>>(mutations.d_mutations_freq, d_freq_index, d_scan_index, offset, demography(pop,0), pop, mutations.h_num_populations, mutations.h_array_Length);
		offset += (int)(ceil((demography(pop,0) - 1)/4.f)*4);
	}
	cudaMemsetAsync(mutations.d_mutations_age,0,mutations.h_array_Length*sizeof(int)); //eventually will replace where mutations have age <= 0 (age before sim start)

	cudaFree(d_freq_index);
	cudaFree(d_scan_index);
}

//assumes prev_sim.num_sites is equivalent to current simulations num_sites or prev_sim.num_mutations == 0 (initialize to blank)
template <typename Functor_mutation, typename Functor_demography>
__host__ __forceinline__ void init_blank_prev_run(sim_struct & mutations, const sim_result & prev_sim, const Functor_mutation mu_rate, const Functor_demography demography, const float num_sites, const int seed, const int compact_rate){
	//if prev_sim.num_mutations == 0 or num sites or num_populations between two runs are not equivalent, don't copy (initialize to blank)
	if(prev_sim.num_mutations != 0 && num_sites == prev_sim.num_sites && mutations.h_num_populations == prev_sim.num_populations){
		set_Index_Length(mutations, prev_sim.num_mutations, mu_rate, demography, num_sites, compact_rate, mutations.h_num_populations, 0);
		cout<<"initial length " << mutations.h_array_Length << endl;
		cudaMalloc((void**)&mutations.d_mutations_freq, mutations.h_num_populations*mutations.h_array_Length*sizeof(float));
		cudaMalloc((void**)&mutations.d_mutations_age, mutations.h_array_Length*sizeof(int));

		cudaMemcpyAsync(mutations.d_mutations_freq, prev_sim.mutations_freq, prev_sim.num_populations*prev_sim.num_mutations*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(mutations.d_mutations_age, prev_sim.mutations_freq, prev_sim.num_mutations*sizeof(int), cudaMemcpyHostToDevice);
	}else{
		set_Index_Length(mutations, 0, mu_rate, demography, num_sites, compact_rate, mutations.h_num_populations, 0);
		cout<<"initial length " << mutations.h_array_Length << endl;
		cudaMalloc((void**)&mutations.d_mutations_freq, mutations.h_num_populations*mutations.h_array_Length*sizeof(float));
		cudaMalloc((void**)&mutations.d_mutations_age, mutations.h_array_Length*sizeof(int));
	}
}

template <typename Functor_mutation, typename Functor_demography>
__host__ __forceinline__ void compact(sim_struct & mutations, const int generation, const Functor_mutation mu_rate, const Functor_demography demography, const float num_sites, const int compact_rate){
	int * d_flag;
	cudaMalloc((void**)&d_flag,mutations.h_mutations_Index*sizeof(int));

	flag_segregating_mutations<<<50,1024>>>(d_flag, mutations.d_mutations_freq, mutations.h_num_populations, mutations.h_mutations_Index, mutations.h_array_Length);

	int * d_scan_Index;
	cudaMalloc((void**)&d_scan_Index, mutations.h_mutations_Index*sizeof(int));

	void * d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_flag, d_scan_Index, mutations.h_mutations_Index);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_flag, d_scan_Index, mutations.h_mutations_Index);
	cudaFree(d_temp_storage);

	int h_num_seg_mutations;
	cudaMemcpy(&h_num_seg_mutations, &d_scan_Index[mutations.h_mutations_Index-1], sizeof(int), cudaMemcpyDeviceToHost);
	h_num_seg_mutations += 1; //doesn't need to be h_num_seg_mutations += d_flag[mutations.h_mutations_Index-1] as last mutation in index will be new, so d_flag[mutations.h_mutations_Index-1] = 1

	int old_mutations_Index = mutations.h_mutations_Index;
	int old_array_Length = mutations.h_array_Length;
	set_Index_Length(mutations, h_num_seg_mutations, mu_rate, demography, num_sites, compact_rate, mutations.h_num_populations, generation);

	float * d_temp;
	int * d_temp2;
	cudaMalloc((void**)&d_temp,mutations.h_num_populations*mutations.h_array_Length*sizeof(float));
	cudaMalloc((void**)&d_temp2,mutations.h_array_Length*sizeof(int));

	const dim3 gridsize(50,mutations.h_num_populations,1);
	scatter_arrays<<<gridsize,1024>>>(d_temp, d_temp2, mutations.d_mutations_freq, mutations.d_mutations_age, d_flag, d_scan_Index, old_mutations_Index, mutations.h_array_Length, old_array_Length);

	cudaFree(mutations.d_mutations_freq);
	cudaFree(mutations.d_mutations_age);

	mutations.d_mutations_freq = d_temp;
	mutations.d_mutations_age = d_temp2;

	cudaFree(d_flag);
	cudaFree(d_scan_Index);
}

template <typename Functor_mutation, typename Functor_demography, typename Functor_inbreeding, typename Functor_migration, typename Functor_selection>
__host__ __forceinline__ sim_result run_sim(const Functor_mutation mu_rate, const Functor_demography demography, const Functor_migration m, const Functor_selection s, const float h, const Functor_inbreeding FI, const float num_sites, const int num_populations, const int seed, const bool init_mse = true, const sim_result & prev_sim = sim_result(), const int compact_rate = 40){
	sim_struct mutations;
	mutations.h_num_populations = num_populations;
	mutations.h_new_mutation_Indices = new int[mutations.h_num_populations+1];

	//----- initialize simulation -----
	if(init_mse){
		//----- mutation-selection equilibrium (mse) (default) -----
		initialize_mse(mutations, mu_rate, demography, s, h, num_sites, seed, compact_rate);
		//----- end -----
	}else{
		//----- initialize from results of previous simulation run or initialize to blank (blank will often take >> N generations to reach equilibrium) -----
		init_blank_prev_run(mutations, prev_sim, mu_rate, demography, num_sites, seed, compact_rate);
		//----- end -----
	}
	//----- end -----

	cout<<"initial num_mutations " << mutations.h_mutations_Index << endl;

	//----- simulation steps -----
	cudaStream_t * pop_streams = new cudaStream_t[mutations.h_num_populations];
	for(int pop = 0; pop < mutations.h_num_populations; pop++){ cudaStreamCreate(&pop_streams[pop]); }
	cudaEvent_t kernelEvent;
	cudaEventCreateWithFlags((&kernelEvent), cudaEventDisableTiming);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int generation = 1;
	while(true){
		int N = demography(0,generation);
		if(N == -1){ break; } //check for end of simulation

		//-----migration, selection, drift -----
		for(int pop = 0; pop < mutations.h_num_populations; pop++){
			N = demography(pop,generation);
			int F = FI(pop,generation);
			migration_selection_drift<<<800,128,0,pop_streams[pop]>>>(mutations.d_mutations_freq, mutations.h_mutations_Index, mutations.h_array_Length, N, m, s, h, F, seed, pop, mutations.h_num_populations, generation);
		}
		//----- end  -----

		//-----generate new mutations -----
		calc_new_mutations_Index(mutations, mu_rate, demography, num_sites, seed, mutations.h_num_populations, generation);
		for(int pop = 0; pop < mutations.h_num_populations; pop++){
			N = demography(pop,generation);
			int prev_Index = mutations.h_new_mutation_Indices[pop];
			int new_Index = mutations.h_new_mutation_Indices[pop+1];
			add_new_mutations<<<5,1024,0,pop_streams[pop]>>>(mutations.d_mutations_freq, mutations.d_mutations_age, prev_Index, new_Index, mutations.h_array_Length, 1.f/N, pop, mutations.h_num_populations, generation);
		}
		mutations.h_mutations_Index = mutations.h_new_mutation_Indices[mutations.h_num_populations];
		//----- end -----

		//-----compact every compact_rate generations and final generation -----
		if((generation % compact_rate == 0) || demography(0,generation+1) == -1){ compact(mutations, generation, mu_rate, demography, num_sites, compact_rate); }
		//----- end -----

		generation++;
		//cout<<"num_mutations " << mutations.h_mutations_Index <<" generations " << generation << endl;
	}
	//----- end -----

	refactor_mutation_age<<<50,1024>>>(mutations.d_mutations_age, mutations.h_mutations_Index, generation);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("time elapsed generations: %f\n", elapsedTime);

	sim_result out(mutations, num_sites, generation);
	for(int pop = 0; pop < mutations.h_num_populations; pop++){ cudaStreamDestroy(pop_streams[pop]); }
	cudaEventDestroy(kernelEvent);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	delete pop_streams;

	return out;
}


int main(int argc, char **argv)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int N_chrom_pop = 2*pow(10.f,4); //constant population for now
    float gamma = 0;
	float s = gamma/(2.f*N_chrom_pop);
	float h = 0.5;
	float F = 1;
	float mu = pow(10.f,-9); //per-site mutation rate
	float L = 2.5*pow(10.f,8); //eventually set so the number of expected mutations is > a certain amount
	float m = 0;
	int num_pop = 3;

	const int total_number_of_generations = pow(10.f,4);
	const int seed = 0xdecafbad;
	demography burn_in(N_chrom_pop,5);
	demography dem(N_chrom_pop,total_number_of_generations);
	inbreeding Fi(F);

	sim_result a = run_sim(mutation(mu), dem, mig_prop(m), sel_coeff(s), h, Fi, L, num_pop, seed);
	cout<<endl<<"final number of mutations: " << a.num_mutations << endl;

/*	sim_result a = run_sim(mutation(mu), burn_in, sel_coeff(s), h, L, seed);
	cout<<endl<<"final number of mutations: " << a.num_mutations << endl;

	sim_result b = run_sim(mutation(mu), dem, sel_coeff(s), h, L, seed+1, false, a);
	cout<<endl<<"final number of mutations: " << b.num_mutations << endl;*/

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("time elapsed: %f\n", elapsedTime);
	cudaDeviceReset();
}
