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

using namespace std;
using namespace cub;
using namespace r123;

#define __DEBUG__ false

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
__host__ __device__ __forceinline__ float uint_float_01(unsigned int in){
	//(mostly) stolen from Philox code "uniform.hpp"
	R123_CONSTEXPR float factor = float(1.)/(UINT_MAX + float(1.));
	R123_CONSTEXPR float halffactor = float(0.5)*factor;
    return in*factor + halffactor;
}

inline __host__ __device__ int4 round(float4 f){ return make_int4(round(f.x), round(f.y), round(f.z), round(f.w)); }

inline __host__ __device__  float4 operator-(float a, float4 b){ return make_float4((a-b.x), (a-b.y), (a-b.z), (a-b.w)); }

__host__ __device__ __forceinline__  uint4 Philox(int2 seed, int k, int step, int population, int round){
	typedef Philox4x32_R<8> P; //can change the 10 rounds of bijection down to 8 (lowest safe limit) to get possible extra speed!
	P rng;

	P::key_type key = {{seed.x, seed.y}}; //random int to set key space + seed
	P::ctr_type count = {{k, step, population, round}};

	union {
		P::ctr_type c;
		uint4 i;
	}u;

	u.c = rng(count, key);

	return u.i;
}

__host__ __device__ __forceinline__ void pois_iter(float j, float mean, float & emu, float & cdf){
	emu *= mean/j;
	cdf += emu;
}

__host__ __device__ __forceinline__ int poiscdfinv(float p, float mean){
	float emu = expf(-1 * mean);
	float cdf = emu;
	if(cdf >= p){ return 0; }

	pois_iter(1.f, mean, emu, cdf); if(cdf >= p){ return 1; }
	pois_iter(2.f, mean, emu, cdf); if(cdf >= p){ return 2; }
	pois_iter(3.f, mean, emu, cdf); if(cdf >= p){ return 3; }
	pois_iter(4.f, mean, emu, cdf); if(cdf >= p){ return 4; }
	pois_iter(5.f, mean, emu, cdf); if(cdf >= p){ return 5; }
	pois_iter(6.f, mean, emu, cdf); if(cdf >= p){ return 6; }
	pois_iter(7.f, mean, emu, cdf); if(cdf >= p){ return 7; }
	pois_iter(8.f, mean, emu, cdf); if(cdf >= p){ return 8; }
	pois_iter(9.f, mean, emu, cdf); if(cdf >= p){ return 9; }
	pois_iter(10.f, mean, emu, cdf); if(cdf >= p){ return 10; }
	pois_iter(11.f, mean, emu, cdf); if(cdf >= p){ return 11; }
	pois_iter(12.f, mean, emu, cdf); if(cdf >= p){ return 12; }
	pois_iter(13.f, mean, emu, cdf); if(cdf >= p){ return 13; }
	pois_iter(14.f, mean, emu, cdf); if(cdf >= p){ return 14; }
	pois_iter(15.f, mean, emu, cdf); if(cdf >= p){ return 15; }
	pois_iter(16.f, mean, emu, cdf); if(cdf >= p){ return 16; }
	pois_iter(17.f, mean, emu, cdf); if(cdf >= p){ return 17; }
	pois_iter(18.f, mean, emu, cdf); if(cdf >= p){ return 18; }
	pois_iter(19.f, mean, emu, cdf); if(cdf >= p){ return 19; }
	pois_iter(20.f, mean, emu, cdf); if(cdf >= p){ return 20; }
	pois_iter(21.f, mean, emu, cdf); if(cdf >= p){ return 21; }
	pois_iter(22.f, mean, emu, cdf); if(cdf >= p){ return 22; }
	pois_iter(23.f, mean, emu, cdf); if(cdf >= p){ return 23; }

	return 24; //limit for mean <= 6, 32 limit for mean <= 10, max float between 0 and 1 is 0.99999999
}

__host__ __device__ __forceinline__ int RandBinom(float p, float N, int2 seed, int k, int step, int population, int start_round){
//only for use when N is small
	int j = 0;
	int counter = 0;
	
	union {
		uint h[4];
		uint4 i;
	}u;
	
	while(counter < N){
		u.i = Philox(seed, k, step, population, start_round+counter);
		
		int counter2 = 0;
		while(counter < N && counter2 < 4){
			if(uint_float_01(u.h[counter2]) <= p){ j++; }
			counter++;
			counter2++;
		}
	}
	
	return j;
}

//faster on home GPU if inlined
__host__ __device__ __forceinline__ int Rand1(float mean, float var, float p, float N, int2 seed, int k, int step, int population){

	if(N <= 50){ return RandBinom(p, N, seed, k, step, population, 0); } //for some reason compiler on home GPU inlines function with this line, but not without it
	uint4 i = Philox(seed, k, step, population, 0);
	if(mean <= 6){ return poiscdfinv(uint_float_01(i.x), mean); }
	else if(mean >= N-6){ return N - poiscdfinv(uint_float_01(i.x), N-mean); } //flip side of binomial, when 1-p is small
	return round(normcdfinv(uint_float_01(i.x))*sqrtf(var)+mean);
}

//faster on home GPU if don't inline!
__device__ __noinline__ int Rand1(unsigned int i, float mean, float var, float N){
	if(mean <= 6){ return poiscdfinv(uint_float_01(i), mean); }
	else if(mean >= N-6){ return N - poiscdfinv(uint_float_01(i), N-mean); } //flip side of binomial, when 1-p is small
	return round(normcdfinv(uint_float_01(i))*sqrtf(var)+mean);
}

//faster on home GPU if inlined
__device__ __forceinline__ int4 Rand4(float4 mean, float4 var, float4 p, float N, int2 seed, int k, int step, int population){
	if(N <= 50){ return make_int4(RandBinom(p.x, N, seed, k, step, population, 0),RandBinom(p.y, N, seed, k, step, population, N),RandBinom(p.z, N, seed, k, step, population, 2*N),RandBinom(p.w, N, seed, k, step, population, 3*N)); }
	uint4 i = Philox(seed, k, step, population, 0);
	return make_int4(Rand1(i.x, mean.x, var.x, N), Rand1(i.y, mean.y, var.y, N), Rand1(i.z, mean.z, var.z, N), Rand1(i.w, mean.w, var.w, N));
}

__device__ __forceinline__ double mse(double i, int N, float F, float h, float s){ //takes in double from mse_integrand, otherwise takes in float
		return exp(2*N*s*i*((2*h+(1-2*h)*i)*(1-F) + 2*F)/(1+F)); //works for either haploid or diploid, N should be number of individuals, for haploid, F = 1
}

template <typename Functor_selection>
struct mse_integrand{
	Functor_selection sel_coeff;
	int N, pop, gen;
	float F, h;

	mse_integrand(): N(0), h(0), F(0), pop(0), gen(0) {}
	mse_integrand(Functor_selection xsel_coeff, int xN, float xF, float xh, int xpop, int xgen = 0): sel_coeff(xsel_coeff), N(xN), F(xF), h(xh), pop(xpop), gen(xgen) { }

	__device__ __forceinline__ double operator()(double i) const{
		float s = sel_coeff(pop, gen, i); //eventually may allow mse function to be set by user so as to allow for mse of frequency-dependent selection
		return mse(i, N, F, h, -1*s); //exponent term in integrand is negative inverse
	}
};


template<typename Functor_function>
struct trapezoidal_upper{
	Functor_function fun;
	trapezoidal_upper() { }
	trapezoidal_upper(Functor_function xfun): fun(xfun) { }
	__device__ __forceinline__ double operator()(double a, double step_size) const{ return step_size*(fun(a)+fun(a-step_size))/2; } //upper integral
};

//generates an array of areas from 1 to 0 of frequencies at every step size
template <typename Functor_Integrator>
__global__ void calculate_area(double * d_freq, const int num_freq, const double step_size, Functor_Integrator trapezoidal){
	int myID = blockIdx.x*blockDim.x + threadIdx.x;

	for(int id = myID; id < num_freq; id += blockDim.x*gridDim.x){ d_freq[id] = trapezoidal((1.0 - id*step_size), step_size); }
}

__global__ void reverse_array(double * array, const int N){
	int myID = blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < N/2; id += blockDim.x*gridDim.x){
		double temp = array[N - id - 1];
		array[N - id - 1] = array[id];
		array[id] = temp;
	}
}

//determines number of mutations at each frequency in the initial population, sets it equal to mutation-selection balance
template <typename Functor_selection>
__global__ void initialize_mse_frequency_array(int * freq_index, double * mse_integral, const int offset, const float mu, const int Nind, const int Nchrom, const float L, const Functor_selection sel_coeff, const float F, const float h, const int2 seed, const int population){
	int myID = blockIdx.x*blockDim.x + threadIdx.x;

	double mse_total = mse_integral[0]; //integral from frequency 0 to 1
	for(int id = myID; id < (Nchrom-1); id += blockDim.x*gridDim.x){ //exclusive, number of freq in pop is chromosome population size N-1
		float i = (id+1.f)/Nchrom;
		float s = sel_coeff(population, 0, i);
		float lambda;
		if(s == 0){ lambda = 2*mu*L/i; }
		else{ lambda = 2*mu*L*(mse(i, Nind, F, h, s)*mse_integral[id])/(mse_total*i*(1-i)); }  //cast to type double4 not allowed, so not using vector memory optimization
		freq_index[offset+id] = max(Rand1(lambda, lambda, mu, L*Nchrom, seed, 0, id, population),0);//round(lambda);// //  //mutations are poisson distributed in each frequency class
	}
}

//fills in mutation array using the freq and scan indices
//y threads correspond to freq_index/scan_index indices, use grid-stride loops
//x threads correspond to mutation array indices, use grid-stride loops
//using scan number to define start of array, freq_index to define num_new_mutations_index (if 0 simply ignore) and myIDx used to calculate allele_count
__global__ void initialize_mse_mutation_array(float * mutations_freq, const int * freq_index, const int * scan_index, const int offset, const int Nchrom, const int population, const int num_populations, const int array_Length){
	int myIDy = blockIdx.y*blockDim.y + threadIdx.y;
	for(int idy = myIDy; idy < (Nchrom-1); idy+= blockDim.y*gridDim.y){
		int myIDx = blockIdx.x*blockDim.x + threadIdx.x;
		int start = scan_index[offset+idy];
		int num_mutations = freq_index[offset+idy];
		float freq = (idy+1.f)/Nchrom;
		for(int idx = myIDx; idx < num_mutations; idx+= blockDim.x*gridDim.x){
			for(int pop = 0; pop < num_populations; pop++){ mutations_freq[pop*array_Length + start + idx] = 0; }
			mutations_freq[population*array_Length + start + idx] = freq;
		}
	}
}

__global__ void mse_set_mutID(int4 * mutations_ID, const float * const mutations_freq, const int mutations_Index, const int num_populations, const int array_Length, const int device){
	int myID = blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < mutations_Index; id+= blockDim.x*gridDim.x){
		for(int pop = 0; pop < num_populations; pop++){
			if(mutations_freq[pop*array_Length+id] > 0){
				mutations_ID[id] = make_int4(0,pop,id,device);//age: eventually will replace where mutations have age <= 0 (age before sim start)//threadID//population
				break; //assumes mutations are only in one population at start
			}
		}
	}
}

/*__global__ void print_Device_array_uint(unsigned int * array, int num){

	for(int i = 0; i < num; i++){
		//if(i%1000 == 0){ printf("\n"); }
		printf("%d: %d\t",i,array[i]);
	}
}

__global__ void sum_Device_array_bit(unsigned int * array, int num){
	//int sum = 0;
	for(int i = 0; i < num; i++){
		//if(i%1000 == 0){ printf("\n"); }
		//unsigned int n = array[i];
		while (n) {
		    if (n & 1)
		    	sum+=1;
		    n >>= 1;
		}
		printf("%d\t",__popc(array[i]));
	}
}

__global__ void sum_Device_array_uint(unsigned int * array, int num){
	int j = 0;
	for(int i = 0; i < num; i++){
		j += array[i];
	}
	printf("%d",j);
}*/

//calculates new frequencies for every mutation in the population
//seed for random number generator philox's key space, id, generation for its counter space in the pseudorandom sequence
template <typename Functor_migration, typename Functor_selection>
__global__ void migration_selection_drift(float * mutations_freq, float * const prev_freq, const int mutations_Index, const int array_Length, const int N, const Functor_migration mig_prop, const Functor_selection sel_coeff, const float F, const float h, const int2 seed, const int population, const int num_populations, const int generation){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;

	for(int id = myID; id < mutations_Index/4; id+= blockDim.x*gridDim.x){
		float4 i_mig = make_float4(0);
		for(int pop = 0; pop < num_populations; pop++){
			float4 i = reinterpret_cast<float4*>(prev_freq)[pop*array_Length/4+id]; //allele frequency in previous population //make sure array length is divisible by 4 (preferably divisible by 32 or warp_size)!!!!!!
			i_mig += mig_prop(pop,population,generation)*i; //if population is size 0 or extinct < this does not protect user if they have an incorrect migration function
		}
		float4 s = make_float4(sel_coeff(population,generation,i_mig.x),sel_coeff(population,generation,i_mig.y),sel_coeff(population,generation,i_mig.z),sel_coeff(population,generation,i_mig.w));
		float4 i_mig_sel = (s*i_mig*i_mig+i_mig+(F+h-h*F)*s*i_mig*(1-i_mig))/(i_mig*i_mig*s+(F+2*h-2*h*F)*s*i_mig*(1-i_mig)+1);
		float4 mean = i_mig_sel*N; //expected allele count in new generation
		int4 j_mig_sel_drift = clamp(Rand4(mean,(1.0-i_mig_sel)*mean,i_mig_sel,N,seed,(id + 2),generation,population), 0, N);
		reinterpret_cast<float4*>(mutations_freq)[population*array_Length/4+id] = make_float4(j_mig_sel_drift)/N; //final allele freq in new generation //make sure array length is divisible by 4 (preferably 32/warp_size)!!!!!!
	}
	int id = myID + mutations_Index/4 * 4;  //only works if minimum of 3 threads are launched
	if(id < mutations_Index){
		float i_mig = 0;
		for(int pop = 0; pop < num_populations; pop++){
			float i = prev_freq[pop*array_Length+id]; //allele frequency in previous population
			i_mig += mig_prop(pop,population,generation)*i;
		}
		float s = sel_coeff(population,generation,i_mig);
		float i_mig_sel = (s*i_mig*i_mig+i_mig+(F+h-h*F)*s*i_mig*(1-i_mig))/(i_mig*i_mig*s+(F+2*h-2*h*F)*s*i_mig*(1-i_mig)+1);
		float mean = i_mig_sel*N; //expected allele count in new generation
		int j_mig_sel_drift = clamp(Rand1(mean,(1.0-i_mig_sel)*mean,i_mig_sel,N,seed,(id + 2),generation,population), 0, N);
		mutations_freq[population*array_Length+id] = float(j_mig_sel_drift)/N; //final allele freq in new generation
	}
}

__global__ void add_new_mutations(float * mutations_freq, int4 * mutations_ID, const int prev_mutations_Index, const int new_mutations_Index, const int array_Length, float freq, const int population, const int num_populations, const int generation, const int device){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; (id < (new_mutations_Index-prev_mutations_Index)) && ((id + prev_mutations_Index) < array_Length); id+= blockDim.x*gridDim.x){
		for(int pop = 0; pop < num_populations; pop++){ mutations_freq[(pop*array_Length+prev_mutations_Index+id)] = 0; }
		mutations_freq[(population*array_Length+prev_mutations_Index+id)] = freq;
		mutations_ID[(prev_mutations_Index+id)] = make_int4(generation,population,id,device);
	}
}

__device__ int boundary_0(float freq){
	return (freq <= 0.f);
}

__device__ int boundary_1(float freq){
	return (freq >= 1.f);
}

//tests indicate accumulating mutations in non-migrating populations is not much of a problem
template <typename Functor_demography>
__global__ void flag_segregating_mutations(unsigned int * flag, unsigned int * counter, const Functor_demography demography, const float * const mutations_freq, const int num_populations, const int padded_mut_Index, const int mutations_Index, const int array_Length, const int generation, const int warp_size){
//adapted from https://www.csuohio.edu/engineering/sites/csuohio.edu.engineering/files/Research_Day_2015_EECS_Poster_14.pdf
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < (padded_mut_Index >> 5); id+= blockDim.x*gridDim.x){
		int lnID = threadIdx.x % warp_size;
		int warpID = id >> 5;

		unsigned int mask;
		unsigned int cnt=0;

		for(int j = 0; j < 32; j++){
			int zero = 1;
			int one = 1;
			int index = (warpID<<10)+(j<<5)+lnID;
			if(index >= mutations_Index){ zero *= 1; one = 0; }
			else{
				for(int pop = 0; pop < num_populations; pop++){
					if(demography(pop,generation) > 0){ //not protected if population goes extinct but demography function becomes non-zero again (shouldn't happen anyway)
						float i = mutations_freq[pop*array_Length+index];
						zero *= boundary_0(i);
						one *= boundary_1(i);
					}
				}
			}
			mask = __ballot(!(zero+one)); //1 if allele is segregating in any population, 0 otherwise

			if(lnID == 0) {
				flag[(warpID<<5)+j] = mask;
				cnt += __popc(mask);
			}
		}

		if(lnID == 0) counter[warpID] = cnt; //store sum
	}
}

__global__ void scatter_arrays(float * new_mutations_freq, int4 * new_mutations_ID, const float * const mutations_freq, const int4 * const mutations_ID, const unsigned int * const flag, const unsigned int * const scan_Index, const int padded_mut_Index, const int new_array_Length, const int old_array_Length, const int warp_size){
//adapted from https://www.csuohio.edu/engineering/sites/csuohio.edu.engineering/files/Research_Day_2015_EECS_Poster_14.pdf
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	int population = blockIdx.y;

	for(int id = myID; id < (padded_mut_Index >> 5); id+= blockDim.x*gridDim.x){
		int lnID = threadIdx.x % warp_size;
		int warpID = id >> 5;

		unsigned int predmask;
		unsigned int cnt;

		predmask = flag[(warpID<<5)+lnID];
		cnt = __popc(predmask);

		//parallel prefix sum
#pragma unroll
		for(int offset = 1; offset < 32; offset<<=1){
			unsigned int n = __shfl_up(cnt, offset);
			if(lnID >= offset) cnt += n;
		}

		unsigned int global_index = 0;
		if(warpID > 0) global_index = scan_Index[warpID - 1];

		for(int i = 0; i < 32; i++){
			unsigned int mask = __shfl(predmask, i); //broadcast from thread i
			unsigned int sub_group_index = 0;
			if(i > 0) sub_group_index = __shfl(cnt, i-1);
			if(mask & (1 << lnID)){
				new_mutations_freq[population*new_array_Length + global_index + sub_group_index + __popc(mask & ((1 << lnID) - 1))] = mutations_freq[population*old_array_Length+(warpID<<10)+(i<<5)+lnID];
				if(population == 0){ new_mutations_ID[global_index + sub_group_index + __popc(mask & ((1 << lnID) - 1))] = mutations_ID[(warpID<<10)+(i<<5)+lnID]; }
			}
		}
	}
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

struct mig_prop_pop
{
	float m;
	int num_pop;
	mig_prop_pop(float m, int n) : m(m), num_pop(n){ }
	__device__ __forceinline__ float operator()(const int pop_FROM, const int pop_TO, const int generation) const{
		if(pop_FROM == pop_TO){ return 1-(num_pop-1)*m; }
		return (num_pop > 1) * m;
	}
};


struct sel_coeff
{
	float s;
	sel_coeff() : s(0) {}
	sel_coeff(float s) : s(s){ }
	__device__ __forceinline__ float operator()(const int population, const int generation, const float freq) const{
		return s;
	}
};

struct demography
{
	int N;
	demography(int N) : N(N){ }
	__host__ __device__ __forceinline__ int operator()(const int population, const int generation) const{
		return N;
	}
};


struct dominance
{
	float h;
	dominance() : h(0) {}
	dominance(float h) : h(h){ }
	__host__ __forceinline__ float operator()(const int population, const int generation) const{
		return h;
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

#define cudaCheckErrors(expr1,expr2,expr3) { cudaError_t e = expr1; int g = expr2; int p = expr3; if (e != cudaSuccess) { fprintf(stderr,"error %d %s\tfile %s\tline %d\tgeneration %d\t population %d\n", e, cudaGetErrorString(e),__FILE__,__LINE__, g,p); exit(1); } }
#define cudaCheckErrorsAsync(expr1,expr2,expr3) { cudaCheckErrors(expr1,expr2,expr3); if(__DEBUG__){ cudaCheckErrors(cudaDeviceSynchronize(),expr2,expr3); } }

//for internal function passing
struct sim_struct{
	//device arrays
	float * d_mutations_freq; //allele frequency of current mutations
	float * d_prev_freq; // meant for storing frequency values so changes in previous populations' frequencies don't affect later populations' migration
	int4 * d_mutations_ID;  //generation mutations appeared in simulation, ID that generated mutation, population that mutation first arose, device simulation was run on

	int h_num_populations; //number of populations (# rows for freq)
	int h_array_Length; //full length of the mutation array, total number of mutations across all populations (# columns for freq)
	int h_mutations_Index; //number of mutations in the population (last mutation is at h_mutations_Index-1)
	int * h_new_mutation_Indices; //indices of new mutations, current age/freq index of mutations is at position 0, index for mutations in population 0 to be added to array is at position 1, etc ...
	bool * h_extinct; //boolean if population has gone extinct
	int warp_size; //device warp size, determines the amount of extra padding on array to allow for memory coalscence

	sim_struct(): h_num_populations(0), h_array_Length(0), h_mutations_Index(0), warp_size(0) { d_mutations_freq = NULL; d_prev_freq = NULL; d_mutations_ID = NULL; h_new_mutation_Indices = NULL; h_extinct = NULL;}

	~sim_struct(){ cudaCheckErrorsAsync(cudaFree(d_mutations_freq),-1,-1); cudaCheckErrorsAsync(cudaFree(d_prev_freq),-1,-1); cudaCheckErrorsAsync(cudaFree(d_mutations_ID),-1,-1); if(h_new_mutation_Indices) { delete [] h_new_mutation_Indices; } if(h_extinct){ delete [] h_extinct; } }
};

struct mutID{
	int generation,population,threadID,device; //generation mutation appeared in simulation, population in which mutation first arose, threadID that generated mutation, device that generated mutation
};

//for final sim result output
struct sim_result{
	float * mutations_freq; //allele frequency of mutations in final generation
	mutID * mutations_ID; //unique ID consisting of generation, population, threadID, and device
	bool * extinct; //extinct[pop] == true, flag if population is extinct by end of simulation
	int num_populations; //number of populations in freq array (array length, rows)
	int num_mutations; //number of mutations in array (array length for age/freq, columns)
	int num_sites; //number of sites in simulation
	int total_generations; //number of generations in the simulation

	sim_result(): num_populations(0), num_mutations(0), num_sites(0), total_generations(0) { mutations_freq = NULL; mutations_ID = NULL; extinct = NULL; }

	static void store_sim_result(sim_result & out, sim_struct & mutations, int num_sites, int total_generations, cudaStream_t * control_streams, cudaEvent_t * control_events){
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

	~sim_result(){ if(mutations_freq){ cudaCheckErrors(cudaFreeHost(mutations_freq),-1,-1); } if(mutations_ID){ cudaCheckErrors(cudaFreeHost(mutations_ID),-1,-1); } if(extinct){ delete [] extinct; } }
};



//for site frequency spectrum output
struct sfs{
	int * frequency_spectrum;
	int ** frequency_age_spectrum;
	int * populations; //which populations are in SFS
	int * num_samples; //number of samples taken for each population
	int num_populations;
	int num_sites;
	int total_generations; //number of generations in the simulation

	sfs(): num_populations(0), num_sites(0), total_generations(0) {frequency_spectrum = NULL; frequency_age_spectrum = NULL; populations = NULL; num_samples = NULL;}
	~sfs(){ if(frequency_spectrum){ delete[] frequency_spectrum; } if(frequency_age_spectrum){ delete[] frequency_age_spectrum; } if(populations){ delete[] populations; } if(num_samples){ delete[] num_samples; }}
};

template <typename Functor_mu, typename Functor_demography, typename Functor_inbreeding>
__host__ __forceinline__ void set_Index_Length(sim_struct & mutations, const int num_mutations, const Functor_mu mu_rate, const Functor_demography demography, const Functor_inbreeding FI, const float num_sites, const int compact_rate, const int generation, const int final_generation){
	mutations.h_mutations_Index = num_mutations;
	mutations.h_array_Length = mutations.h_mutations_Index;
	for(int gen = generation+1; gen <= (generation+compact_rate) && gen <= final_generation; gen++){
		for(int pop = 0; pop < mutations.h_num_populations; pop++){
			int Nchrom_e = 2*demography(pop,generation)/(1+FI(pop,generation));
			if(Nchrom_e == 0 || mutations.h_extinct[pop]){ continue; }
			mutations.h_array_Length += mu_rate(pop,gen)*Nchrom_e*num_sites + 7*sqrtf(mu_rate(pop,gen)*Nchrom_e*num_sites); //maximum distance of floating point normal rng is <7 stdevs from mean
		}
	}

	mutations.h_array_Length = (int)(mutations.h_array_Length/mutations.warp_size + 1*(mutations.h_array_Length%mutations.warp_size!=0))*mutations.warp_size; //extra padding for coalesced global memory access, /watch out: one day warp sizes may not be multiples of 4
}

template <typename Functor_mutation, typename Functor_demography, typename Functor_inbreeding>
__host__ __forceinline__ void calc_new_mutations_Index(sim_struct & mutations, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_inbreeding FI, const float L, const int2 seed, const int generation){
	int num_new_mutations = 0;
	mutations.h_new_mutation_Indices[0] = mutations.h_mutations_Index;
	for(int pop = 0; pop < mutations.h_num_populations; pop++){
		int Nchrom_e = 2*demography(pop,generation)/(1+FI(pop,generation));
		if(Nchrom_e == 0 || mutations.h_extinct[pop]){ continue; }
		float mu = mu_rate(pop, generation);
		float lambda = mu*Nchrom_e*L;
		int temp = max(Rand1(lambda, lambda, mu, Nchrom_e*L, seed, 1, generation, pop),0);
		num_new_mutations += temp;
		mutations.h_new_mutation_Indices[pop+1] = num_new_mutations + mutations.h_mutations_Index;
	}
}

template <typename Functor_selection>
__host__ __forceinline__ void integrate_mse(double * d_mse_integral, const int N_ind, const int Nchrom_e, const Functor_selection sel_coeff, const float F, const float h, int pop, cudaStream_t pop_stream){
	double * d_freq;

	cudaCheckErrorsAsync(cudaMalloc((void**)&d_freq, Nchrom_e*sizeof(double)),0,pop);

	mse_integrand<Functor_selection> mse_fun(sel_coeff, N_ind, F, h, pop);
	trapezoidal_upper< mse_integrand<Functor_selection> > trap(mse_fun);

	calculate_area<<<10,1024,0,pop_stream>>>(d_freq, Nchrom_e, (double)1.0/(Nchrom_e), trap); //setup array frequency values to integrate over (upper integral from 1 to 0)
	cudaCheckErrorsAsync(cudaPeekAtLastError(),0,pop);

	void * d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	cudaCheckErrorsAsync(DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_freq, d_mse_integral, Nchrom_e, pop_stream),0,pop);
	cudaCheckErrorsAsync(cudaMalloc(&d_temp_storage, temp_storage_bytes),0,pop);
	cudaCheckErrorsAsync(DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_freq, d_mse_integral, Nchrom_e, pop_stream),0,pop);
	cudaCheckErrorsAsync(cudaFree(d_temp_storage),0,pop);
	cudaCheckErrorsAsync(cudaFree(d_freq),0,pop);

	reverse_array<<<10,1024,0,pop_stream>>>(d_mse_integral, Nchrom_e);
	cudaCheckErrorsAsync(cudaPeekAtLastError(),0,pop);
}

template <typename Functor_mutation, typename Functor_demography, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance>
__host__ __forceinline__ void initialize_mse(sim_struct & mutations, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_selection sel_coeff, const Functor_inbreeding FI, const Functor_dominance dominance, const int final_generation, const float num_sites, const int2 seed, const int compact_rate, cudaStream_t * pop_streams, cudaEvent_t * pop_events, const int myDevice){

	int num_freq = 0; //number of frequencies
	for(int pop = 0; pop < mutations.h_num_populations; pop++){
		float F = FI(pop,0);
		int Nchrom_e = 2*demography(pop,0)/(1+F);
		if(Nchrom_e <= 1){ continue; }
		num_freq += (Nchrom_e - 1);
	}

	int * d_freq_index;
	cudaCheckErrorsAsync(cudaMalloc((void**)&d_freq_index, num_freq*sizeof(int)),0,-1);
	int * d_scan_index;
	cudaCheckErrorsAsync(cudaMalloc((void**)&d_scan_index, num_freq*sizeof(int)),0,-1);
	double ** mse_integral = new double *[mutations.h_num_populations];

	int offset = 0;
	for(int pop = 0; pop < mutations.h_num_populations; pop++){
		int N_ind = demography(pop,0);
		float mu = mu_rate(pop,0);
		float F = FI(pop,0);
		int Nchrom_e = 2*N_ind/(1+F);
		float h = dominance(pop,0);
		cudaCheckErrorsAsync(cudaMalloc((void**)&mse_integral[pop], Nchrom_e*sizeof(double)),0,pop);
		if(Nchrom_e <= 1){ continue; }
		integrate_mse(mse_integral[pop], N_ind, Nchrom_e, sel_coeff, F, h, pop, pop_streams[pop]);
		initialize_mse_frequency_array<<<6,1024,0,pop_streams[pop]>>>(d_freq_index, mse_integral[pop], offset, mu, N_ind, Nchrom_e, num_sites, sel_coeff, F, h, seed, pop);
		cudaCheckErrorsAsync(cudaPeekAtLastError(),0,pop);
		offset += (Nchrom_e - 1);
	}

	for(int pop = 0; pop < mutations.h_num_populations; pop++){
		cudaCheckErrorsAsync(cudaFree(mse_integral[pop]),0,pop);
		cudaCheckErrorsAsync(cudaEventRecord(pop_events[pop],pop_streams[pop]),0,pop);
		cudaCheckErrorsAsync(cudaStreamWaitEvent(pop_streams[0],pop_events[pop],0),0,pop);
	}

	delete [] mse_integral;

	void * d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	cudaCheckErrorsAsync(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_freq_index, d_scan_index, num_freq, pop_streams[0]),0,-1);
	cudaCheckErrorsAsync(cudaMalloc(&d_temp_storage, temp_storage_bytes),0,-1);
	cudaCheckErrorsAsync(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_freq_index, d_scan_index, num_freq, pop_streams[0]),0,-1);
	cudaCheckErrorsAsync(cudaFree(d_temp_storage),0,-1);

	cudaCheckErrorsAsync(cudaEventRecord(pop_events[0],pop_streams[0]),0,-1);
	for(int pop = 0; pop < mutations.h_num_populations; pop++){
		cudaCheckErrorsAsync(cudaStreamWaitEvent(pop_streams[pop],pop_events[0],0),0,pop);
	}

	int prefix_sum_result;
	int final_freq_count;
	//final index is numfreq-1
	cudaCheckErrors(cudaMemcpy(&prefix_sum_result, &d_scan_index[(num_freq-1)], sizeof(int), cudaMemcpyDeviceToHost),0,-1); //has to be in sync with host as result is used straight afterwards
	cudaCheckErrors(cudaMemcpy(&final_freq_count, &d_freq_index[(num_freq-1)], sizeof(int), cudaMemcpyDeviceToHost),0,-1); //has to be in sync with host as result is used straight afterwards
	int num_mutations = prefix_sum_result+final_freq_count;
	set_Index_Length(mutations, num_mutations, mu_rate, demography, FI, num_sites, compact_rate, 0, final_generation);
	//cout<<"initial length " << mutations.h_array_Length << endl;

	cudaCheckErrorsAsync(cudaMalloc((void**)&mutations.d_mutations_freq, mutations.h_num_populations*mutations.h_array_Length*sizeof(float)),0,-1);
	cudaCheckErrorsAsync(cudaMalloc((void**)&mutations.d_prev_freq, mutations.h_num_populations*mutations.h_array_Length*sizeof(float)),0,-1);
	cudaCheckErrorsAsync(cudaMalloc((void**)&mutations.d_mutations_ID, mutations.h_array_Length*sizeof(int4)),0,-1);

	const dim3 blocksize(4,256,1);
	const dim3 gridsize(32,32,1);
	offset = 0;
	for(int pop = 0; pop < mutations.h_num_populations; pop++){
		float F = FI(pop,0);
		int Nchrom_e = 2*demography(pop,0)/(1+F);
		if(Nchrom_e <= 1){ continue; }
		initialize_mse_mutation_array<<<gridsize,blocksize,0,pop_streams[pop]>>>(mutations.d_prev_freq, d_freq_index, d_scan_index, offset, Nchrom_e, pop, mutations.h_num_populations, mutations.h_array_Length);
		cudaCheckErrorsAsync(cudaPeekAtLastError(),0,pop);
		offset += (Nchrom_e - 1);
	}

	mse_set_mutID<<<50,1024,0,pop_streams[mutations.h_num_populations]>>>(mutations.d_mutations_ID, mutations.d_prev_freq, mutations.h_mutations_Index, mutations.h_num_populations, mutations.h_array_Length, myDevice);
	cudaCheckErrorsAsync(cudaPeekAtLastError(),0,-1);

	for(int pop = 0; pop <= mutations.h_num_populations; pop++){
		cudaCheckErrorsAsync(cudaEventRecord(pop_events[pop],pop_streams[pop]),0,pop);
		for(int pop2 = mutations.h_num_populations+1; pop2 < 2*mutations.h_num_populations; pop2++){
			cudaCheckErrorsAsync(cudaStreamWaitEvent(pop_streams[pop2],pop_events[pop],0),0,pop); //tells every pop_stream not used above to wait until initialization is done
		}
	}

	cudaCheckErrorsAsync(cudaFree(d_freq_index),0,-1);
	cudaCheckErrorsAsync(cudaFree(d_scan_index),0,-1);
}

//assumes prev_sim.num_sites is equivalent to current simulations num_sites or prev_sim.num_mutations == 0 (initialize to blank)
template <typename Functor_mutation, typename Functor_demography, typename Functor_inbreeding>
__host__ __forceinline__ void init_blank_prev_run(sim_struct & mutations, int & generation_shift, int & final_generation, const sim_result & prev_sim, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_inbreeding FI, const float num_sites, const int compact_rate, cudaStream_t * pop_streams, cudaEvent_t * pop_events){
	//if prev_sim.num_mutations == 0 or num sites or num_populations between two runs are not equivalent, don't copy (initialize to blank)
	int num_mutations = 0;
	bool use_prev_sim = (num_sites == prev_sim.num_sites && mutations.h_num_populations == prev_sim.num_populations);
	if(prev_sim.num_mutations != 0 && use_prev_sim){
		generation_shift = prev_sim.total_generations;
		final_generation += generation_shift;
		num_mutations = prev_sim.num_mutations;
		for(int i = 0; i < mutations.h_num_populations; i++){ mutations.h_extinct[i] = prev_sim.extinct[i]; }
	}

		set_Index_Length(mutations, num_mutations, mu_rate, demography, FI, num_sites, compact_rate, generation_shift, final_generation);
		//cout<<"initial length " << mutations.h_array_Length << endl;
		cudaCheckErrorsAsync(cudaMalloc((void**)&mutations.d_mutations_freq, mutations.h_num_populations*mutations.h_array_Length*sizeof(float)),0,-1);
		cudaCheckErrorsAsync(cudaMalloc((void**)&mutations.d_prev_freq, mutations.h_num_populations*mutations.h_array_Length*sizeof(float)),0,-1);
		cudaCheckErrorsAsync(cudaMalloc((void**)&mutations.d_mutations_ID, mutations.h_array_Length*sizeof(int3)),0,-1);

	if(prev_sim.num_mutations != 0 && use_prev_sim){
		cudaCheckErrorsAsync(cudaMemcpy2DAsync(mutations.d_prev_freq, mutations.h_array_Length*sizeof(float), prev_sim.mutations_freq, prev_sim.num_mutations*sizeof(float), prev_sim.num_mutations*sizeof(float), prev_sim.num_populations, cudaMemcpyHostToDevice, pop_streams[0]),0,-1);
		cudaCheckErrorsAsync(cudaMemcpyAsync(mutations.d_mutations_ID, prev_sim.mutations_ID, prev_sim.num_mutations*sizeof(int4), cudaMemcpyHostToDevice, pop_streams[1]),0,-1);

		cudaCheckErrorsAsync(cudaEventRecord(pop_events[0],pop_streams[0]),0,-1);
		cudaCheckErrorsAsync(cudaEventRecord(pop_events[1],pop_streams[1]),0,-1);

		//wait until initialization is complete
		for(int pop = 2; pop < 2*mutations.h_num_populations; pop++){
			cudaCheckErrorsAsync(cudaStreamWaitEvent(pop_streams[pop],pop_events[0],0),0,pop);
			cudaCheckErrorsAsync(cudaStreamWaitEvent(pop_streams[pop],pop_events[1],0),0,pop);
		}
	}
}

template <typename Functor_mutation, typename Functor_demography, typename Functor_inbreeding>
__host__ __forceinline__ void compact(sim_struct & mutations, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_inbreeding FI, const float num_sites, const int generation, const int final_generation, const int compact_rate, cudaStream_t * control_streams, cudaEvent_t * control_events, cudaStream_t * pop_streams){
	unsigned int * d_flag;
	unsigned int * d_count;

	int padded_mut_index = (((mutations.h_mutations_Index>>10)+1*(mutations.h_mutations_Index%1024!=0))<<10);

	cudaCheckErrorsAsync(cudaFree(mutations.d_mutations_freq),generation,-1);
	cudaCheckErrorsAsync(cudaMalloc((void**)&d_flag,(padded_mut_index>>5)*sizeof(unsigned int)),generation,-1);
	cudaCheckErrorsAsync(cudaMalloc((void**)&d_count,(padded_mut_index>>10)*sizeof(unsigned int)),generation,-1);

	flag_segregating_mutations<<<800,128,0,control_streams[0]>>>(d_flag, d_count, demography, mutations.d_prev_freq, mutations.h_num_populations, padded_mut_index, mutations.h_mutations_Index, mutations.h_array_Length, generation, mutations.warp_size);
	cudaCheckErrorsAsync(cudaPeekAtLastError(),generation,-1);

	unsigned int * d_scan_Index;
	cudaCheckErrorsAsync(cudaMalloc((void**)&d_scan_Index,(padded_mut_index>>10)*sizeof(unsigned int)),generation,-1);

	void * d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	cudaCheckErrorsAsync(DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_count, d_scan_Index, (padded_mut_index>>10), control_streams[0]),generation,-1);
	cudaCheckErrorsAsync(cudaMalloc(&d_temp_storage, temp_storage_bytes),generation,-1);
	cudaCheckErrorsAsync(DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_count, d_scan_Index, (padded_mut_index>>10), control_streams[0]),generation,-1);
	cudaCheckErrorsAsync(cudaFree(d_temp_storage),generation,-1);

	cudaCheckErrorsAsync(cudaPeekAtLastError(),generation,-1);

	int h_num_seg_mutations;
	cudaCheckErrors(cudaMemcpy(&h_num_seg_mutations, &d_scan_Index[(padded_mut_index>>10)-1], sizeof(int), cudaMemcpyDeviceToHost),generation,-1); //has to be in sync with the host since h_num_seq_mutations is manipulated on CPU right after

	int old_array_Length = mutations.h_array_Length;
	set_Index_Length(mutations, h_num_seg_mutations, mu_rate, demography, FI, num_sites, compact_rate, generation, final_generation);

	float * d_temp;
	int4 * d_temp2;
	cudaCheckErrorsAsync(cudaMalloc((void**)&d_temp,mutations.h_num_populations*mutations.h_array_Length*sizeof(float)),generation,-1);
	cudaCheckErrorsAsync(cudaMalloc((void**)&d_temp2,mutations.h_array_Length*sizeof(int4)),generation,-1);

	const dim3 gridsize(800,mutations.h_num_populations,1);
	scatter_arrays<<<gridsize,128,0,control_streams[0]>>>(d_temp, d_temp2, mutations.d_prev_freq, mutations.d_mutations_ID, d_flag, d_scan_Index, padded_mut_index, mutations.h_array_Length, old_array_Length, mutations.warp_size);
	cudaCheckErrorsAsync(cudaPeekAtLastError(),generation,-1);

	cudaCheckErrorsAsync(cudaEventRecord(control_events[0],control_streams[0]),generation,-1);

	for(int pop = 0; pop < 2*mutations.h_num_populations; pop++){
		cudaCheckErrorsAsync(cudaStreamWaitEvent(pop_streams[pop],control_events[0],0),generation,pop);
	}

	cudaCheckErrorsAsync(cudaStreamWaitEvent(control_streams[1],control_events[0],0),generation,-1);
	cudaCheckErrorsAsync(cudaStreamWaitEvent(control_streams[2],control_events[0],0),generation,-1);

	cudaCheckErrorsAsync(cudaFree(mutations.d_prev_freq),generation,-1);
	cudaCheckErrorsAsync(cudaFree(mutations.d_mutations_ID),generation,-1);

	mutations.d_prev_freq = d_temp;
	mutations.d_mutations_ID = d_temp2;

	cudaCheckErrorsAsync(cudaFree(d_flag),generation,-1);
	cudaCheckErrorsAsync(cudaFree(d_scan_Index),generation,-1);
	cudaCheckErrorsAsync(cudaFree(d_count),generation,-1);

	cudaCheckErrorsAsync(cudaMalloc((void**)&mutations.d_mutations_freq,mutations.h_num_populations*mutations.h_array_Length*sizeof(float)),generation,-1);
}

__host__ __forceinline__ void swap_freq_pointers(sim_struct & mutations){
	float * temp = mutations.d_prev_freq;
	mutations.d_prev_freq = mutations.d_mutations_freq;
	mutations.d_mutations_freq = temp;
}

struct no_sample{
	__host__ __forceinline__ int operator()(const int generation) const{ return 0; }
};

template <typename Functor_mutation, typename Functor_demography, typename Functor_migration, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance, typename Functor_timesample>
__host__ __forceinline__ sim_result * run_sim(const Functor_mutation mu_rate, const Functor_demography demography, const Functor_migration mig_prop, const Functor_selection sel_coeff, const Functor_inbreeding FI, const Functor_dominance dominance, const int num_generations, const float num_sites, const int num_populations, const int seed1, const int seed2, Functor_timesample take_sample, int max_samples = 0, const bool init_mse = true, const sim_result & prev_sim = sim_result(), const int compact_rate = 35, const int cuda_device = -1){
	int2 seed;
	seed.x = seed1;
	seed.y = seed2;

	int cudaDeviceCount;
	cudaCheckErrorsAsync(cudaGetDeviceCount(&cudaDeviceCount),-1,-1);
	if(cuda_device >= 0 && cuda_device < cudaDeviceCount){ cudaCheckErrors(cudaSetDevice(cuda_device),-1,-1); } //unless user specifies, driver auto-magically selects free GPU to run on
	int myDevice;
	cudaCheckErrorsAsync(cudaGetDevice(&myDevice),-1,-1);
	cudaDeviceProp devProp;
	cudaCheckErrors(cudaGetDeviceProperties(&devProp, myDevice),-1,-1);

	sim_struct mutations;
	sim_result * all_results = new sim_result[max_samples+1];

	mutations.h_num_populations = num_populations;
	mutations.h_new_mutation_Indices = new int[mutations.h_num_populations+1];
	mutations.h_extinct = new bool[mutations.h_num_populations];
	for(int i = 0; i < mutations.h_num_populations; i++){ mutations.h_extinct[i] = 0; } //some compilers won't default to 0
	mutations.warp_size  = devProp.warpSize;

	cudaStream_t * pop_streams = new cudaStream_t[2*mutations.h_num_populations];
	cudaEvent_t * pop_events = new cudaEvent_t[2*mutations.h_num_populations];

	int num_control_streams = 4;
	cudaStream_t * control_streams = new cudaStream_t[num_control_streams];;
	cudaEvent_t * control_events = new cudaEvent_t[num_control_streams];;

	for(int pop = 0; pop < 2*mutations.h_num_populations; pop++){
		cudaCheckErrors(cudaStreamCreate(&pop_streams[pop]),-1,pop);
		cudaCheckErrorsAsync(cudaEventCreateWithFlags(&pop_events[pop],cudaEventDisableTiming),-1,pop);
	}

	for(int stream = 0; stream < num_control_streams; stream++){
		cudaCheckErrors(cudaStreamCreate(&control_streams[stream]),-1,stream);
		cudaCheckErrorsAsync(cudaEventCreateWithFlags(&control_events[stream],cudaEventDisableTiming),-1,stream);
	}

	int generation = 0;
	int final_generation = num_generations;

	//----- initialize simulation -----
	if(init_mse){
		//----- mutation-selection equilibrium (mse) (default) -----
		initialize_mse(mutations, mu_rate, demography, sel_coeff, FI, dominance, final_generation, num_sites, seed, compact_rate, pop_streams, pop_events, myDevice);
		//----- end -----
	}else{
		//----- initialize from results of previous simulation run or initialize to blank (blank will often take >> N generations to reach equilibrium) -----
		init_blank_prev_run(mutations, generation, final_generation, prev_sim, mu_rate, demography, FI, num_sites, compact_rate, pop_streams, pop_events);
		//----- end -----
	}
	//----- end -----

	//cout<< endl <<"initial num_mutations " << mutations.h_mutations_Index;

	//----- simulation steps -----

	int next_compact_generation = generation + compact_rate;
	int sample_index = 0;

	while((generation+1) <= final_generation){ //end of simulation
		generation++;
		//----- migration, selection, drift -----
		for(int pop = 0; pop < mutations.h_num_populations; pop++){
			int N_ind = demography(pop,generation);
			if(mutations.h_extinct[pop]){ continue; }
			if(N_ind <= 0){
				if(demography(pop,generation-1) > 0){ //previous generation, the population was alive
					N_ind = 0; //allow to go extinct
					mutations.h_extinct[pop] = true; //next generation will not process
				} else{ continue; } //if population has not yet arisen, it will have a population size of 0, can simply not process
			}
			float F = FI(pop,generation);
			int Nchrom_e = 2*N_ind/(1+F);
			float h = dominance(pop,generation);
			//10^5 mutations: 600 blocks for 1 population, 300 blocks for 3 pops
			migration_selection_drift<<<600,128,0,pop_streams[pop]>>>(mutations.d_mutations_freq, mutations.d_prev_freq, mutations.h_mutations_Index, mutations.h_array_Length, Nchrom_e, mig_prop, sel_coeff, F, h, seed, pop, mutations.h_num_populations, generation);
			cudaCheckErrorsAsync(cudaPeekAtLastError(),generation,pop);
			cudaCheckErrorsAsync(cudaEventRecord(pop_events[pop],pop_streams[pop]),generation,pop);
		}
		//----- end -----

		//----- generate new mutations -----
		calc_new_mutations_Index(mutations, mu_rate, demography, FI, num_sites, seed, generation);
		for(int pop = 0; pop < mutations.h_num_populations; pop++){
			int N_ind = demography(pop,generation);
			if((N_ind <= 0) || mutations.h_extinct[pop]){ continue; }
			float F = FI(pop,generation);
			int Nchrom_e = 2*N_ind/(1+F);
			float freq = 1.f/Nchrom_e;
			int prev_Index = mutations.h_new_mutation_Indices[pop];
			int new_Index = mutations.h_new_mutation_Indices[pop+1];
			add_new_mutations<<<20,512,0,pop_streams[pop+mutations.h_num_populations]>>>(mutations.d_mutations_freq, mutations.d_mutations_ID, prev_Index, new_Index, mutations.h_array_Length, freq, pop, mutations.h_num_populations, generation, myDevice);
			cudaCheckErrorsAsync(cudaPeekAtLastError(),generation,pop);
			cudaCheckErrorsAsync(cudaEventRecord(pop_events[pop+mutations.h_num_populations],pop_streams[pop+mutations.h_num_populations]),generation,pop);
		}
		mutations.h_mutations_Index = mutations.h_new_mutation_Indices[mutations.h_num_populations];
		//----- end -----

		for(int pop1 = 0; pop1 < mutations.h_num_populations; pop1++){
			for(int pop2 = 0; pop2 < mutations.h_num_populations; pop2++){
				cudaCheckErrorsAsync(cudaStreamWaitEvent(pop_streams[pop1],pop_events[pop2+mutations.h_num_populations],0), generation, pop1*mutations.h_num_populations+pop2); //wait to do the next round of mig_sel_drift until
			}
			if((take_sample(generation) && sample_index < max_samples && take_sample(generation) > 1) || (generation == next_compact_generation || generation == final_generation)){
				cudaCheckErrorsAsync(cudaStreamWaitEvent(control_streams[0],pop_events[pop1],0), generation, pop1); //wait to compact/or record data until after mig_sel_drift and add_new_mut are done
				cudaCheckErrorsAsync(cudaStreamWaitEvent(control_streams[0],pop_events[pop1+mutations.h_num_populations],0), generation, pop1);
			}
			if((take_sample(generation) && sample_index < max_samples)){
				cudaCheckErrorsAsync(cudaStreamWaitEvent(control_streams[1],pop_events[pop1],0), generation, pop1);
				cudaCheckErrorsAsync(cudaStreamWaitEvent(control_streams[1],pop_events[pop1+mutations.h_num_populations],0), generation, pop1);
				cudaCheckErrorsAsync(cudaStreamWaitEvent(control_streams[2],pop_events[pop1],0), generation, pop1);
				cudaCheckErrorsAsync(cudaStreamWaitEvent(control_streams[2],pop_events[pop1+mutations.h_num_populations],0), generation, pop1);
			}
		}
		if(cudaStreamQuery(control_streams[1]) != cudaSuccess){ cudaCheckErrors(cudaStreamSynchronize(control_streams[1]), generation, -1); } //if not yet done streaming to host from previous generation, pause here
		if(cudaStreamQuery(control_streams[2]) != cudaSuccess){ cudaCheckErrors(cudaStreamSynchronize(control_streams[2]), generation, -1); }

		swap_freq_pointers(mutations);


		//----- take time samples of frequency spectrum -----
		if(take_sample(generation) && sample_index < max_samples){
			//----- compact before sampling if requested -----
			//when testing this, check if I need to swap the mutations freq pointers to make sure compact doesn't screw up which mutation array needs to be where (especially if 2 compacts are performed in a row: sample + compact_rate)
			if(take_sample(generation) > 1){ compact(mutations, mu_rate, demography, FI, num_sites, generation, final_generation, compact_rate, control_streams, control_events, pop_streams); next_compact_generation = generation + compact_rate; }
			//----- end -----
			sim_result::store_sim_result(all_results[sample_index], mutations, num_sites, generation, control_streams, control_events);
			sample_index++;
		}
		//----- end -----

		//----- compact every compact_rate generations and final generation -----
		if(generation == next_compact_generation || generation == final_generation){ compact(mutations, mu_rate, demography, FI, num_sites, generation, final_generation, compact_rate, control_streams, control_events, pop_streams); next_compact_generation = generation + compact_rate;  }
		//----- end -----
	}
	//----- end -----

	//----- store final (compacted) generation on host -----
	sim_result::store_sim_result(all_results[max_samples], mutations, num_sites, generation, control_streams, control_events);
	//----- end -----

	if(cudaStreamQuery(control_streams[1]) != cudaSuccess){ cudaCheckErrors(cudaStreamSynchronize(control_streams[1]), generation, -1); }; //wait for writes to host to finish
	if(cudaStreamQuery(control_streams[2]) != cudaSuccess){ cudaCheckErrors(cudaStreamSynchronize(control_streams[2]), generation, -1); };

	for(int pop = 0; pop < 2*mutations.h_num_populations; pop++){ cudaCheckErrorsAsync(cudaStreamDestroy(pop_streams[pop]),generation,pop); cudaCheckErrorsAsync(cudaEventDestroy(pop_events[pop]),generation,pop); }
	for(int stream = 0; stream < num_control_streams; stream++){ cudaCheckErrorsAsync(cudaStreamDestroy(control_streams[stream]),generation,stream); cudaCheckErrorsAsync(cudaEventDestroy(control_events[stream]),generation,stream); }

	delete [] pop_streams;
	delete [] pop_events;
	delete [] control_streams;
	delete [] control_events;

	return all_results;
}

struct neutral
{
	__device__ __forceinline__ float operator()(const int population, const int generation, const float freq) const{ return 0; }
};

struct no_mig
{
	__device__ __forceinline__ float operator()(const int pop_FROM, const int pop_TO, const int generation) const{ return pop_FROM == pop_TO; }
};

__host__ __forceinline__ sim_result sequencing_sample(sim_result sim, int * population, int * num_samples, const int seed){
	//neutral neu;
	//no_mig mig;
	//migration_selection_drift(float * mutations_freq, const int mutations_Index, const int array_Length, const int N, const Functor_migration mig_prop, const Functor_selection sel_coeff, const float h, const float F, const int seed, const int population, const int num_populations, const int generation);
	//compact(sim_struct & mutations, const int generation, const Functor_mutation mu_rate, const Functor_demography demography, const float num_sites, const int compact_rate)
	return sim_result();
}

//multi-population sfs
__host__ __forceinline__ sfs site_frequency_spectrum(sim_result sim, int * population, int * num_samples, int num_sfs_populations, const int seed){
	sim_result samp = sequencing_sample(sim, population, num_samples, seed);

	return sfs();
}

//multi-time point, multi-population sfs
__host__ __forceinline__ sfs temporal_site_frequency_spectrum(sim_result sim, int * population, int * num_samples, int num_sfs_populations, const int seed){
	sim_result samp = sequencing_sample(sim, population, num_samples, seed);

	return sfs();
}

//trace frequency trajectories of mutations from generation start to generation end in a (sub-)population
//can track an individual mutation or groups of mutation by specifying when the mutation was "born", in which population, with what threadID
__host__ __forceinline__ float ** trace_mutations(sim_result * sim, int generation_start, int generation_end, int population, int generation_born = -1, int population_born = -1, int threadID = -1){

	return NULL;
}

int main(int argc, char **argv)
{

    float gamma = 0;
	float h = 0.5;
	float F = 1.0;
	int N_ind = pow(10.f,5)*(1+F); //constant population for now
	float s = gamma/(2*N_ind);
	float mu = pow(10.f,-9); //per-site mutation rate
	int total_number_of_generations = pow(10.f,5);//36;//50;//
	float L = 1*2*pow(10.f,7);
	float m = 0.00;
	int num_pop = 1;
	const int seed2 = 0xdecafbad;
	const int seed1 = 0xbeeff00d;

	sim_result * a = run_sim(mutation(mu), demography(N_ind), mig_prop_pop(m,num_pop), sel_coeff(s), inbreeding(F), dominance(h), total_number_of_generations, L, num_pop, seed1, seed2, no_sample(), 0, true);
	cout<<endl<<"final number of mutations: " << a[0].num_mutations << endl;
	delete [] a;

	a = run_sim(mutation(mu), demography(N_ind), mig_prop_pop(m,num_pop), sel_coeff(s), inbreeding(F), dominance(h), total_number_of_generations, L, num_pop, seed1, seed2, no_sample(), 0, true);
	delete [] a;


    cudaEvent_t start, stop;
    float elapsedTime;
    int num_iter = 10;
    int compact_rate = 35;
    num_pop = 1;
    m = 0.0;

    total_number_of_generations = pow(10.f,3);
    L = 1*2*pow(10.f,7);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for(int i = 0; i < num_iter; i++){
		sim_result * b = run_sim(mutation(mu), demography(N_ind), mig_prop_pop(m,num_pop), sel_coeff(s), inbreeding(F), dominance(h), total_number_of_generations, L, num_pop, seed1, seed2, no_sample(), 0, true, sim_result(), compact_rate);
		if(i==0){ cout<<endl<<"final number of mutations: " << b[0].num_mutations << endl; }
		delete [] b;
	}

	elapsedTime = 0;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("time elapsed: %f\n\n", elapsedTime/num_iter);

	cudaDeviceSynchronize();
	cudaDeviceReset();
}
