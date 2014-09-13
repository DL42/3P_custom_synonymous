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
__global__ void initialize_mse_frequency_array(int * freq_index, const float mu, const int N, const float L, const Functor_selection sel_coeff, const float h, const int seed){
	//determines number of mutations at each frequency in the initial population, sets it equal to mutation-selection balance
	int myID = blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < (N-1)/4; id+= blockDim.x*gridDim.x){ //exclusive, length of freq array is chromosome population size N-1
		float4 i = make_float4((4*id + 1),(4*id + 2),(4*id + 3),(4*id + 4))/N;
		float s = sel_coeff(0, 0.5); //the equations below don't work for frequency-dependent selection anyway
		float4 lambda;
		if(s == 0){ lambda = 2*mu*L/i; }
		else{ lambda =  2*mu*L*((-1.0*expd(-1*(2*N*s)*(-1.f*i+1.0))+1.0)/((-1*exp(-1*(2*N*double(s)))+1)*i*(-1.0*i+1.0))); }
		reinterpret_cast<int4*>(freq_index)[id] = max(Rand4(lambda, lambda, make_float4(mu), L*N, 0, id, seed, 0),make_int4(0)); //round(lambda);//// ////mutations are poisson distributed in each frequency class
		//printf("%d %d %f %f %f %f %f %f %f %f \r", myID, id, i.x, i.y, i.z, i.w, lambda.x, lambda.y, lambda.z, lambda.w);
	}


	int id = myID + (N-1)/4*4; //all integers //right now only works if minimum of 3 threads are launched
	if(id < (N-1)){
		float i = (id+1.f)/N;
		float s = sel_coeff(0, 0.5);
		float lambda;
		if(s == 0){ lambda = 2*mu*L/i; }
		else{ lambda =  2*mu*L*(1-exp(-1*(2*N*double(s))*(1-i)))/((1-exp(-1*(2*N*double(s))))*i*(1-i)); }
		freq_index[id] = max(Rand1(lambda, lambda, mu, L*N, 0, id, seed, 0),0);//round(lambda);// //  //mutations are poisson distributed in each frequency class
		//printf("%d %d %f %f\r", myID, id, i, lambda);
	}
}

__global__ void initialize_mse_mutation_array(float * mutations, const int * freq_index, const int * scan_index, const int N){
	//fills in mutation array using the freq and scan indices
	//y threads correspond to freq_index/scan_index indices, use grid-stride loops
	//x threads correspond to mutation array indices, use grid-stride loops
	//using scan number to define start of array, freq_index to define num_new_mutations_index (if 0 simply ignore) and myIDx used to calculate allele_count
	int myIDy = blockIdx.y*blockDim.y + threadIdx.y;
	for(int idy = myIDy; idy < (N-1); idy+= blockDim.y*gridDim.y){
		int myIDx = blockIdx.x*blockDim.x + threadIdx.x;
		int start = scan_index[idy];
		int num_mutations = freq_index[idy];
		float freq = (idy+1.f)/N;
		for(int idx = myIDx; idx < num_mutations; idx+= blockDim.x*gridDim.x){ mutations[start + idx] = freq; }
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
template <typename Functor_selection>
__global__ void selection_drift(float * mutations, const int mutations_Index, const int N, const Functor_selection sel_coeff, const float h, const float F, const int seed, const int generation){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;

	for(int id = myID; id < mutations_Index/4; id+= blockDim.x*gridDim.x){
		float4 i = reinterpret_cast<float4*>(mutations)[id]; //allele frequency in previous population
		float4 s = make_float4(sel_coeff(generation, i.x),sel_coeff(generation, i.y),sel_coeff(generation, i.z),sel_coeff(generation, i.w));
		float4 i_next = (s*i*i+i+(F+h-h*F)*s*i*(1-i))/(i*i*s+(F+2*h-2*h*F)*s*i*(1-i)+1);
		float4 mean = i_next*N; //expected allele count in new generation
		int4 j = clamp(Rand4(mean,(-1.f*i_next + 1.0)*mean,i_next,N,(id + 2),generation,seed,0), 0, N);
		reinterpret_cast<float4*>(mutations)[id] = make_float4(j)/N; //final allele freq in new generation
	}
	int id = myID + mutations_Index/4 * 4;  //right now only works if minimum of 3 threads are launched
	if(id < mutations_Index){
		float i = mutations[id]; //allele frequency in previous population
		float s = sel_coeff(generation, i);
		float i_next = (s*i*i+i+(F+h-h*F)*s*i*(1-i))/(i*i*s+(F+2*h-2*h*F)*s*i*(1-i)+1);
		float mean = i_next*N; //expected allele count in new generation
		int j = clamp(Rand1(mean,(1.0-i_next)*mean,i_next,N,(id + 2),generation,seed,0), 0, N);
		mutations[id] = float(j)/N; //final allele freq in new generation
	}
}

__global__ void add_new_mutations(float * mutations_freq, int * mutations_age, const int mutations_Index, const int new_mutations_Index, const int array_length, float freq, int generation){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; (id < (new_mutations_Index-mutations_Index)) && ((id + mutations_Index) < array_length); id+= blockDim.x*gridDim.x){ mutations_freq[(mutations_Index+id)] = freq; mutations_age[(mutations_Index+id)] = generation;}
}

__device__ int4 boundary(float4 freq){
	return make_int4((freq.x > 0.f && freq.x < 1.f), (freq.y > 0.f && freq.y < 1.f), (freq.z > 0.f && freq.z < 1.f), (freq.w > 0.f && freq.w < 1.f));
}

__device__ int boundary(float freq){
	return (freq > 0.f && freq < 1.f);
}

__global__ void flag_segregating_mutations(int * flag, const float * const mutations, const int mutations_Index){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < mutations_Index/4; id+= blockDim.x*gridDim.x){
		reinterpret_cast<int4*>(flag)[id] = boundary(reinterpret_cast<const float4*>(mutations)[id]);
	}
	int id = myID + mutations_Index/4 * 4;  //right now only works if minimum of 3 threads are launched
	if(id < mutations_Index){ flag[id] = boundary(mutations[id]); }
}

__global__ void scatter_arrays(float * new_mutations_freq, int * new_mutations_age, const float * const mutations_freq, const int * const mutations_age, const int * const flag, const int * const scan_Index, const int mutations_Index){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;

	for(int id = myID; id < mutations_Index; id+= blockDim.x*gridDim.x){
		if(flag[id]){
			int index = scan_Index[id];
			new_mutations_freq[index] = mutations_freq[id];
			new_mutations_age[index] = mutations_age[id];
		}
	}
}

__global__ void copy_arrays(const float * f_smaller_array, float * f_larger_array, const int * i_smaller_array, int * i_larger_array, int mutations_Index){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < mutations_Index/4; id+= blockDim.x*gridDim.x){
		reinterpret_cast<float4*>(f_larger_array)[id] = reinterpret_cast<const float4*>(f_smaller_array)[id];
		reinterpret_cast<int4*>(i_larger_array)[id] = reinterpret_cast<const int4*>(i_smaller_array)[id];
	}
	int id = myID + mutations_Index/4 * 4;  //right now only works if minimum of 3 threads are launched
	if(id < mutations_Index){
		f_larger_array[id] = f_smaller_array[id];
		i_larger_array[id] = i_smaller_array[id];
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


struct sel_coeff
{
	float s;
	sel_coeff(float s) : s(s){ }
	__device__ __forceinline__ float operator()(const int generation, const float freq) const{
		return s;
	}
};

struct demography
{
	int N;
	int total_generations;
	demography(int N, int total_generations) : N(N), total_generations(total_generations){ }
	__host__ __forceinline__ int operator()(const int generation) const{
		if(generation < total_generations){ return N; }
		return -1;
	}
};

struct inbreeding
{
	float F;
	inbreeding(float F) : F(F){ }
	__host__ __forceinline__ float operator()(const int generation) const{
		return F;
	}
};

struct mutation
{
	float mu;
	mutation(float mu) : mu(mu){ }
	__host__ __forceinline__ float operator()(const int generation) const{
		return mu;
	}
};

//for internal function passing
struct sim_struct{
	//device arrays
	float * d_mutations_freq; //allele frequency of current mutations
	int * d_mutations_age;  //generation when mutation entered the population

	int h_array_length; //full length of the mutation array
	int h_mutations_Index; //number of mutations in the population (last mutation is at h_mutations_Index-1)
	int h_new_mutations_Index; //number of mutations in the population in the new generation (after new mutations enter population(s))

	sim_struct(): h_array_length(0), h_mutations_Index(0), h_new_mutations_Index(0){ d_mutations_freq = NULL; d_mutations_age = NULL; }
	~sim_struct(){ cudaFree(d_mutations_freq); cudaFree(d_mutations_age); }
};

//for final result output
struct sim_result{
	float * mutations_freq; //allele frequency of mutations in final generation
	int * mutations_age; //allele age of mutations in final generation (0 most recent, negative values for older mutations)
	int num_mutations; //number of mutations in array (array length)
	int num_sites; //number of sites in simulation
	int total_generations; //number of generations in the simulation

	sim_result() : num_mutations(0), num_sites(0), total_generations(0) { mutations_freq = NULL; mutations_age = NULL; }
	sim_result(sim_struct & mutations, int num_sites, int total_generations) : num_mutations(mutations.h_mutations_Index), num_sites(num_sites), total_generations(total_generations){
		mutations_freq = new float[num_mutations];
		cudaMemcpyAsync(mutations_freq, mutations.d_mutations_freq, num_mutations*sizeof(float), cudaMemcpyDeviceToHost);
		mutations_age = new int[num_mutations];
		cudaMemcpy(mutations_age, mutations.d_mutations_age, num_mutations*sizeof(int), cudaMemcpyDeviceToHost);
	}
	~sim_result(){ if(mutations_freq){ delete mutations_freq; } if(mutations_age){ delete mutations_age; } }
};

template <typename Functor_mu, typename Functor_dem>
__host__ __forceinline__ void set_Index_Length(sim_struct & mutations, const int num_mutations, const Functor_mu mu_rate, const Functor_dem demography, const float num_sites, const int compact_rate, const int generation){
	mutations.h_mutations_Index = num_mutations;
	mutations.h_array_length = mutations.h_mutations_Index;
	for(int i = generation; i < (generation+compact_rate); i++){
		if(demography(i) == -1){ break; } //population has ended
		mutations.h_array_length += mu_rate(i)*demography(i)*num_sites + 7*sqrtf(mu_rate(i)*demography(i)*num_sites);
	}
}

__host__ __forceinline__ void calc_new_mutations_Index(sim_struct & mutations, const float mu, const int N, const float L, const int seed, const int generation){
	float lambda = mu*N*L;
	int num_new_mutations = max(Rand1(lambda, lambda, mu, N*L, 1, generation, seed, 0),0);
	mutations.h_new_mutations_Index = num_new_mutations + mutations.h_mutations_Index;
}

template <typename Functor_mutation, typename Functor_demography, typename Functor_selection>
__host__ __forceinline__ void initialize_mse(sim_struct & mutations, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_selection s, const int h, const float num_sites, const int seed, const int compact_rate){
	int N = demography(0);
	float mu = mu_rate(0);

	int * freq_index;
	cudaMalloc((void**)&freq_index, (N-1)*sizeof(int));
	int * scan_index;
	cudaMalloc((void**)&scan_index,(N-1)*sizeof(int));

	initialize_mse_frequency_array<<<6,1024>>>(freq_index, mu, N, num_sites, s, h, seed);

	void * d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, freq_index, scan_index, (N-1));
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, freq_index, scan_index, (N-1));
	cudaFree(d_temp_storage);

	int prefix_sum_result;
	int final_freq_count;
	//final index is N-2 (N-1 terms)
	cudaMemcpy(&prefix_sum_result, &scan_index[(demography(0)-2)], sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&final_freq_count, &freq_index[(demography(0)-2)], sizeof(int), cudaMemcpyDeviceToHost);
	int num_mutations = prefix_sum_result+final_freq_count;

	set_Index_Length(mutations, num_mutations, mu_rate, demography, num_sites, compact_rate, 0);
	cout<<"initial length " << mutations.h_array_length << endl;

	cudaMalloc((void**)&mutations.d_mutations_freq, mutations.h_array_length*sizeof(float));
	cudaMalloc((void**)&mutations.d_mutations_age, mutations.h_array_length*sizeof(int));

	const dim3 blocksize(4,256,1);
	const dim3 gridsize(16,32,1);
	initialize_mse_mutation_array<<<gridsize,blocksize>>>(mutations.d_mutations_freq, freq_index, scan_index, N);
	cudaMemsetAsync(mutations.d_mutations_age,0,mutations.h_array_length*sizeof(int)); //eventually will replace where mutations have age <= 0 (age before sim start)

	cudaFree(freq_index);
	cudaFree(scan_index);
}

//assumes prev_sim.num_sites is equivalent to current simulations num_sites or prev_sim.num_mutations == 0 (initialize to blank)
template <typename Functor_mutation, typename Functor_demography>
__host__ __forceinline__ void init_blank_prev_run(sim_struct & mutations, const sim_result & prev_sim, const Functor_mutation mu_rate, const Functor_demography demography, const float num_sites, const int seed, const int compact_rate){
	int N = demography(0);
	float mu = mu_rate(0);

	set_Index_Length(mutations, prev_sim.num_mutations, mu_rate, demography, num_sites, compact_rate, 0);
	cout<<"initial length " << mutations.h_array_length << endl;
	cudaMalloc((void**)&mutations.d_mutations_freq, mutations.h_array_length*sizeof(float));
	cudaMalloc((void**)&mutations.d_mutations_age, mutations.h_array_length*sizeof(int));

	//if prev_sim.num_mutations == 0 or num sites between two runs are not equivalent, don't copy (initialize to blank)
	if(prev_sim.num_mutations != 0 && num_sites == prev_sim.num_sites){
		cudaMemcpyAsync(mutations.d_mutations_freq, prev_sim.mutations_freq, prev_sim.num_mutations*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(mutations.d_mutations_age, prev_sim.mutations_freq, prev_sim.num_mutations*sizeof(int), cudaMemcpyHostToDevice);
	}
}

template <typename Functor_mutation, typename Functor_demography>
__host__ __forceinline__ void compact(sim_struct & mutations, const int generation, const Functor_mutation mu_rate, const Functor_demography demography, const float num_sites, const int compact_rate){
	int * flag;
	cudaMalloc((void**)&flag,mutations.h_mutations_Index*sizeof(int));

	flag_segregating_mutations<<<50,1024>>>(flag, mutations.d_mutations_freq, mutations.h_mutations_Index);

	int * scan_Index;
	cudaMalloc((void**)&scan_Index, mutations.h_mutations_Index*sizeof(int));

	void * d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, flag, scan_Index, mutations.h_mutations_Index);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, flag, scan_Index, mutations.h_mutations_Index);
	cudaFree(d_temp_storage);

	int h_num_seg_mutations;
	cudaMemcpy(&h_num_seg_mutations, &scan_Index[mutations.h_mutations_Index-1], sizeof(int), cudaMemcpyDeviceToHost);
	h_num_seg_mutations += 1;

	int old_mutations_Index = mutations.h_mutations_Index;
	set_Index_Length(mutations, h_num_seg_mutations, mu_rate, demography, num_sites, compact_rate, generation);

	float * temp;
	int * temp2;
	cudaMalloc((void**)&temp,mutations.h_array_length*sizeof(float));
	cudaMalloc((void**)&temp2,mutations.h_array_length*sizeof(int));

	scatter_arrays<<<50,1024>>>(temp, temp2, mutations.d_mutations_freq, mutations.d_mutations_age, flag, scan_Index, old_mutations_Index);

	cudaFree(mutations.d_mutations_freq);
	cudaFree(mutations.d_mutations_age);

	mutations.d_mutations_freq = temp;
	mutations.d_mutations_age = temp2;

	cudaFree(flag);
	cudaFree(scan_Index);
}

template <typename Functor_mutation, typename Functor_demography, typename Functor_inbreeding, typename Functor_selection>
__host__ __forceinline__ sim_result run_sim(const Functor_mutation mu_rate, const Functor_demography demography, const Functor_selection s, const float h, const Functor_inbreeding FI, const float num_sites, const int seed, const bool init_mse = true, const sim_result & prev_sim = sim_result(), const int compact_rate = 40){
	sim_struct mutations;
	int N = demography(0);
	float mu = mu_rate(0);
	float F = FI(0);
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
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1); cudaStreamCreate(&stream2);
	cudaEvent_t kernelEvent;
	cudaEventCreateWithFlags((&kernelEvent), cudaEventDisableTiming);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int generation = 1;
	while(true){
		N = demography(generation);
		if(N == -1){ break; } //end of simulation
		mu = mu_rate(generation);
		F = FI(generation);

		//-----selection & drift -----
		selection_drift<<<1000,64>>>(mutations.d_mutations_freq, mutations.h_mutations_Index, N, s, h, F, seed, generation);
		//----- end -----

		//-----generate new mutations -----
		calc_new_mutations_Index(mutations, mu, N, num_sites, seed, generation);
		add_new_mutations<<<5,1024>>>(mutations.d_mutations_freq, mutations.d_mutations_age, mutations.h_mutations_Index, mutations.h_new_mutations_Index, mutations.h_array_length, 1.f/N, generation);
		mutations.h_mutations_Index = mutations.h_new_mutations_Index;
		//----- end -----

		//-----compact every compact_rate generations and final generation -----
		if((generation % compact_rate == 0) || demography(generation+1) == -1){
			compact(mutations, generation, mu_rate, demography, num_sites, compact_rate);
		}
		//----- end -----
		generation++;
	}
	//----- end -----

	refactor_mutation_age<<<50,1024>>>(mutations.d_mutations_age, mutations.h_mutations_Index, generation);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("time elapsed generations: %f\n", elapsedTime);

	sim_result out(mutations, num_sites, generation);
	cudaStreamDestroy(stream1); cudaStreamDestroy(stream2);
	cudaEventDestroy(kernelEvent);
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

	const int total_number_of_generations = pow(10.f,4);
	const int seed = 0xdecafbad;
	demography burn_in(N_chrom_pop,5);
	demography dem(N_chrom_pop,total_number_of_generations);
	inbreeding Fi(F);
	sim_result a = run_sim(mutation(mu), dem, sel_coeff(s), h, Fi, L, seed);

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
