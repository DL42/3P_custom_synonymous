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

inline __host__ __device__ float4 exp(float4 f){ return make_float4(exp(f.x),exp(f.y),exp(f.z),exp(f.w)); }

inline __host__ __device__ double4 expd(float4 d){ return make_double4(exp(double(d.x)),exp(double(d.y)),exp(double(d.z)),exp(double(d.w))); }

inline __host__ __device__  double4 operator*(int a, double4 b){ return make_double4(a * b.x, a * b.y, a * b.z, a * b.w); }

inline __host__ __device__  double4 operator*(double a, float4 b){ return make_double4(a * b.x, a * b.y, a * b.z, a * b.w); }

inline __host__ __device__  double4 operator*(double4 a, double4 b){ return make_double4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }

inline __host__ __device__  double4 operator+(double4 b, double a){ return make_double4(a + b.x, a + b.y, a + b.z, a + b.w); }

inline __host__ __device__  float4 operator/(double4 a, double4 b){ return make_float4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w); }

inline __host__ __device__  float4 operator-(float a, float4 b){ return make_float4((a-b.x), (a-b.y), (a-b.z), (a-b.w)); }

__host__ __device__ __forceinline__  uint4 Philox(int k, int step, int seed, int population, int round){
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



__host__ __device__ __forceinline__ int poiscdfinv(float p, float mean){
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

__host__ __device__ __forceinline__ int RandBinom(float p, float N, int k, int step, int seed, int population, int start_round){
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

__host__ __device__ __forceinline__ int Rand1(float mean, float var, float p, float N, int k, int step, int seed, int population){

	if(N <= 50){ return RandBinom(p, N, k, step, seed, population, 0); }
	uint4 i = Philox(k, step, seed, population, 0);
	if(mean <= 10){ return poiscdfinv(uint_float_01(i.x), mean); }
	else if(mean >= N-10){ return N - poiscdfinv(uint_float_01(i.x), N-mean); } //flip side of binomial, when 1-p is small
	return round(normcdfinv(uint_float_01(i.x))*sqrtf(var)+mean);
}

__device__ __forceinline__ int Rand1(unsigned int i, float mean, float var, float N){
	if(mean <= 10){ return poiscdfinv(uint_float_01(i), mean); }
	else if(mean >= N-10){ return N - poiscdfinv(uint_float_01(i), N-mean); } //flip side of binomial, when 1-p is small
	return round(normcdfinv(uint_float_01(i))*sqrtf(var)+mean);
}

__device__ __forceinline__ int4 Rand4(float4 mean, float4 var, float4 p, float N, int k, int step, int seed, int population){
	if(N <= 50){ return make_int4(RandBinom(p.x, N, k, step, seed, population, 0),RandBinom(p.y, N, k, step, seed, population, N),RandBinom(p.z, N, k, step, seed, population, 2*N),RandBinom(p.w, N, k, step, seed, population, 3*N)); }
	uint4 i = Philox(k, step, seed, population, 0);
	return make_int4(Rand1(i.x, mean.x, var.x, N), Rand1(i.y, mean.y, var.y, N), Rand1(i.z, mean.z, var.z, N), Rand1(i.w, mean.w, var.w, N));
}

__device__ __forceinline__ double haploid(float i, int N, float s){
		return (1-exp((double)(-1*(2*N*s)*(1-i))));
}

__device__ __forceinline__ double4 haploid(float4 i, int N, float s){
		return (-1.0*expd(-1*(2*N*s)*(-1.f*i+1.0))+1.0);
}

__device__ __forceinline__ double diploid(double i, int N, float h, float s){ //takes in double from diploid_integrand, otherwise takes in float
		return exp(N*s*i*(2*h+(1-2*h)*i));
}

__device__ __forceinline__ double4 diploid(float4 i, int N, float h, float s){
		return expd(N*s*i*(((1-2*h)*i)+2*h));
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
	mse_integrand(Functor_selection xsel_coeff, int xN, float xF, float xh, int xpop, int xgen = 0): N(xN), F(xF), h(xh), pop(xpop), gen(xgen) { sel_coeff = xsel_coeff; }

	__device__ __forceinline__ double operator()(double i) const{
		float s = sel_coeff(pop, gen, 0.5); //not meant to be used for frequency-dependent selection
		return mse(i, N, F, h, -1*s); //exponent term in integrand is negative inverse
	}
};


template<typename Functor_function>
struct trapezoidal_upper{
	Functor_function fun;
	trapezoidal_upper() { }
	trapezoidal_upper(Functor_function xfun) { fun = xfun; }
	__device__ __forceinline__ double operator()(double a, double step_size) const{ return step_size*(fun(a)+fun(a-step_size))/2; } //upper integral
};

//generates an array of frequencies from 1 to 0 of frequencies at every step size
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

/*//determines number of mutations at each frequency in the initial population, sets it equal to mutation-selection balance
template <typename Functor_selection>
__global__ void initialize_mse_frequency_array(int * freq_index, double * diploid_integral, const int offset, const float mu, const int N, const float L, const Functor_selection sel_coeff, const float h, const float F, const int seed, const int population){
	int myID = blockIdx.x*blockDim.x + threadIdx.x;

	for(int id = myID; id < (N-1)/4; id += blockDim.x*gridDim.x){ //exclusive, number of freq in pop is chromosome population size N-1
		float4 i = make_float4((4*id + 1),(4*id + 2),(4*id + 3),(4*id + 4))/N;
		float s = sel_coeff(population, 0, 0.5); //the equations below don't work for frequency-dependent selection anyway
		float4 lambda;
		if(s == 0){ lambda = 2*mu*L/i; }
		else{
			lambda =  2*mu*L*(haploid(i,N,s)/(haploid(0,N,s)*i*(-1.0*i+1.0)));
			if(F < 1 && h != 0.5){
				float4 lambda_dip = 2*mu*L*diploid(i, N, h, s)*reinterpret_cast<double4*>(diploid_integral)[id]/(diploid_integral[0]*i*(-1.0*i+1.0));
				lambda = lambda*F + (1-F)*lambda_dip;
			}
		}
		reinterpret_cast<int4*>(freq_index)[offset/4 + id] = max(Rand4(lambda, lambda, make_float4(mu), L*N, 0, id, seed, population),make_int4(0)); //round(lambda);//// ////mutations are poisson distributed in each frequency class
		printf(" %d %d %d %f %f %f %f %f %f %f %f \r", myID, id, population, i.x, i.y, i.z, i.w, lambda.x, lambda.y, lambda.z, lambda.w);
	}

	int next_offset = (int)(ceil((N-1)/4.f)*4);
	int id = myID + (N-1)/4*4; //all integers //right now only works if minimum of 3 threads are launched
	if(id < (N-1)){
		float i = (id+1.f)/N;
		float s = sel_coeff(population, 0, 0.5);
		float lambda;
		if(s == 0){ lambda = 2*mu*L/i; }
		else{
			lambda =  2*mu*L*haploid(i,N,s)/(haploid(0,N,s)*i*(1-i));
			if(F < 1 && h != 0.5){
				float lambda_dip = 2*mu*L*diploid(i, N, h, s)*diploid_integral[id]/(diploid_integral[0]*i*(1-i));
				lambda = lambda*F + (1-F)*lambda_dip;
			}
		}
		freq_index[offset+id] = max(Rand1(lambda, lambda, mu, L*N, 0, id, seed, population),0);//round(lambda);// //  //mutations are poisson distributed in each frequency class
		printf(" %d %d %d %f %f\r", myID, id, population, i, lambda);
	}else if(id >= (N-1) && id < next_offset){ freq_index[offset+id] = 0; } //ensures padding at end of population is set to 0
}*/

//determines number of mutations at each frequency in the initial population, sets it equal to mutation-selection balance
template <typename Functor_selection>
__global__ void initialize_mse_frequency_array(int * freq_index, double * mse_integral, const int offset, const float mu, const int Nind, const int Nchrom, const float L, const Functor_selection sel_coeff, const float F, const float h, const int seed, const int population){
	int myID = blockIdx.x*blockDim.x + threadIdx.x;

	for(int id = myID; id < (Nchrom-1); id += blockDim.x*gridDim.x){ //exclusive, number of freq in pop is chromosome population size N-1
		float i = (id+1.f)/Nchrom;
		float s = sel_coeff(population, 0, 0.5);
		float lambda;
		if(s == 0){ lambda = 2*mu*L/i; }
		else{
			//if(F == 1){ lambda = 2*mu*L*haploid(i,Nind,s)/(haploid(0,Nind,s)*i*(1-i)); }
			//else{ lambda = 2*mu*L*mse(i, Nind, F, h, s)*mse_integral[id]/(mse_integral[0]*i*(1-i)); }
			//if(F == 1){ s *= 2; }
			lambda = 2*mu*L*(mse(i, Nind, F, h, s)*mse_integral[id])/(mse_integral[0]*i*(1-i));
		}
		freq_index[offset+id] = max(Rand1(lambda, lambda, mu, L*Nchrom, 0, id, seed, population),0);//round(lambda);// //  //mutations are poisson distributed in each frequency class
		//printf(" %d %d %d %f %f %f %f %f\r", myID, id, population, i, lambda, mse(i, Nind, F, h, s), mse_integral[id], mse_integral[0]);
	}

	int next_offset = (int)(ceil((Nchrom-1)/4.f)*4);
	int id = myID + (Nchrom-1)/4*4; //all integers //right now only works if minimum of 3 threads are launched
	if(id >= (Nchrom-1) && id < next_offset){ freq_index[offset+id] = 0; } //ensures padding at end of population is set to 0
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
				mutations_ID[id] = make_int4(0,id,pop,device);//age: eventually will replace where mutations have age <= 0 (age before sim start)//threadID//population
				break; //assumes mutations are only in one population at start
			}
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
__global__ void migration_selection_drift(float * mutations_freq, float * const prev_freq, const int mutations_Index, const int array_Length, const int N, const Functor_migration mig_prop, const Functor_selection sel_coeff, const float F, const float h, const int seed, const int population, const int num_populations, const int generation){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;

	for(int id = myID; id < mutations_Index/4; id+= blockDim.x*gridDim.x){
		float4 i_mig = make_float4(0);
		for(int pop = 0; pop < num_populations; pop++){
			float4 i = reinterpret_cast<float4*>(prev_freq)[pop*array_Length/4+id]; //allele frequency in previous population //make sure array length is divisible by 4 (preferably divisible by 32 or warp_size)!!!!!!
			i_mig += mig_prop(pop,population,generation)*i;
		}
		float4 s = make_float4(sel_coeff(population,generation,i_mig.x),sel_coeff(population,generation,i_mig.y),sel_coeff(population,generation,i_mig.z),sel_coeff(population,generation,i_mig.w));
		float4 i_mig_sel = (s*i_mig*i_mig+i_mig+(F+h-h*F)*s*i_mig*(1-i_mig))/(i_mig*i_mig*s+(F+2*h-2*h*F)*s*i_mig*(1-i_mig)+1);
		float4 mean = i_mig_sel*N; //expected allele count in new generation
		int4 j_mig_sel_drift = clamp(Rand4(mean,(-1.f*i_mig_sel + 1.0)*mean,i_mig_sel,N,(id + 2),generation,seed,population), 0, N);
		reinterpret_cast<float4*>(mutations_freq)[population*array_Length/4+id] = make_float4(j_mig_sel_drift)/N; //final allele freq in new generation //make sure array length is divisible by 4 (preferably 32/warp_size)!!!!!!
	}
	int id = myID + mutations_Index/4 * 4;  //only works if minimum of 3 threads are launched
	if(id < mutations_Index){
		float i_mig = 0;
		for(int pop = 0; pop < num_populations; pop++){
			float i = prev_freq[pop*array_Length+id]; //allele frequency in previous population
			i_mig += mig_prop(pop,population,generation)*i;
		}
		float s = sel_coeff(0,generation,i_mig);
		float i_mig_sel = (s*i_mig*i_mig+i_mig+(F+h-h*F)*s*i_mig*(1-i_mig))/(i_mig*i_mig*s+(F+2*h-2*h*F)*s*i_mig*(1-i_mig)+1);
		float mean = i_mig_sel*N; //expected allele count in new generation
		int j_mig_sel_drift = clamp(Rand1(mean,(1.0-i_mig_sel)*mean,i_mig_sel,N,(id + 2),generation,seed,population), 0, N);
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
	int id = myID + mutations_Index/4 * 4;  //only works if minimum of 3 threads are launched
	if(id < mutations_Index){
		int i = 1;
		for(int pop = 0; pop < num_populations; pop++){ i *= boundary(mutations_freq[pop*array_Length+id]); }
		flag[id] = !i; //1 if allele is segregating in any population, 0 otherwise
	}
}

__global__ void scatter_arrays(float * new_mutations_freq, int4 * new_mutations_ID, const float * const mutations_freq, const int4 * const mutations_ID, const int * const flag, const int * const scan_Index, const int mutations_Index, const int new_array_Length, const int old_array_Length){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	int population = blockIdx.y;
	for(int id = myID; id < mutations_Index; id+= blockDim.x*gridDim.x){
		if(flag[id]){
			int index = scan_Index[id];
			new_mutations_freq[population*new_array_Length+index] = mutations_freq[population*old_array_Length+id];
			if(population == 0){ new_mutations_ID[index] = mutations_ID[id]; }
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
	__host__ __forceinline__ int operator()(const int population, const int generation) const{
		return N;
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
	float * d_prev_freq; // meant for storing frequency values so changes in previous populations' frequencies don't affect later populations' migration
	int4 * d_mutations_ID;  //generation mutations appeared in simulation, ID that generated mutation, population that mutation first arose, device simulation was run on

	int h_num_populations; //number of populations (# rows for freq)
	int h_array_Length; //full length of the mutation array, total number of mutations across all populations (# columns for freq)
	int h_mutations_Index; //number of mutations in the population (last mutation is at h_mutations_Index-1)
	int * h_new_mutation_Indices; //indices of new mutations, current age/freq index of mutations is at position 0, index for mutations in population 0 to be added to array is at position 1, etc ...
	bool * extinct; //boolean if population has gone extinct
	int warp_size; //device warp size, determines the amount of extra padding on array to allow for memory coalscence

	sim_struct(): h_num_populations(0), h_array_Length(0), h_mutations_Index(0), warp_size(0) { d_mutations_freq = NULL; d_prev_freq = NULL; d_mutations_ID = NULL; h_new_mutation_Indices = NULL; extinct = NULL;}

	~sim_struct(){ cudaFree(d_mutations_freq); cudaFree(d_prev_freq); cudaFree(d_mutations_ID); if(h_new_mutation_Indices) { delete [] h_new_mutation_Indices; } if(extinct){ delete [] extinct; } }
};

struct mutID{
	int generation,population,threadID,device; //generation mutations appeared in simulation, population in which mutation first arose, ID that generated mutation, Device that generated mutation
};

//for final sim result output
struct sim_result{
	float * mutations_freq; //allele frequency of mutations in final generation
	mutID * mutations_ID; //unique ID consisting of generation, threadID, and population
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
		cudaMallocHost((void**)&out.mutations_freq,out.num_populations*out.num_mutations*sizeof(float)); //should allow for simultaneous transfer to host
		cudaMemcpy2DAsync(out.mutations_freq, out.num_mutations*sizeof(float), mutations.d_prev_freq, mutations.h_array_Length*sizeof(float), out.num_mutations*sizeof(float), out.num_populations, cudaMemcpyDeviceToHost, control_streams[1]); //removes padding
		cudaMallocHost((void**)&out.mutations_ID, out.num_mutations*sizeof(mutID));
		cudaMemcpyAsync(out.mutations_ID, mutations.d_mutations_ID, out.num_mutations*sizeof(int4), cudaMemcpyDeviceToHost, control_streams[2]); //mutations array is 1D
		out.extinct = new bool[out.num_populations];
		for(int i = 0; i < out.num_populations; i++){ out.extinct[i] = mutations.extinct[i]; }

		cudaEventRecord(control_events[1],control_streams[1]);
		cudaEventRecord(control_events[2],control_streams[2]);
		cudaStreamWaitEvent(control_streams[0],control_events[1],0); //if compacting is about to happen, don't compact until results are compiled
		cudaStreamWaitEvent(control_streams[0],control_events[2],0);
		//1 round of migration_selection_drift and add_new_mutations can be done simultaneously with above as they change d_mutations_freq array, not d_prev_freq
	}

	~sim_result(){ if(mutations_freq){ cudaFree(mutations_freq); } if(mutations_ID){ cudaFree(mutations_ID); } if(extinct){ delete [] extinct; } }
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

template <typename Functor_mu, typename Functor_dem, typename Functor_inbreeding>
__host__ __forceinline__ void set_Index_Length(sim_struct & mutations, const int num_mutations, const Functor_mu mu_rate, const Functor_dem demography, const Functor_inbreeding FI, const float num_sites, const int compact_rate, const int generation, const int final_generation){
	mutations.h_mutations_Index = num_mutations;
	mutations.h_array_Length = mutations.h_mutations_Index;
	for(int gen = generation+1; gen <= (generation+compact_rate) && gen < final_generation; gen++){
		for(int pop = 0; pop < mutations.h_num_populations; pop++){
			int Nchrom_e = 2*demography(pop,generation)/(1+FI(pop,generation));
			if(Nchrom_e == 0 || mutations.extinct[pop]){ continue; }
			mutations.h_array_Length += mu_rate(pop,gen)*Nchrom_e*num_sites + 7*sqrtf(mu_rate(pop,gen)*Nchrom_e*num_sites); //maximum distance of floating point normal rng is <7 stdevs from mean
		}
	}
	mutations.h_array_Length = (int)(ceil(mutations.h_array_Length/((float)mutations.warp_size))*mutations.warp_size); //extra padding for coalesced global memory access
}

template <typename Functor_mutation, typename Functor_demography, typename Functor_inbreeding>
__host__ __forceinline__ void calc_new_mutations_Index(sim_struct & mutations, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_inbreeding FI, const float L, const int seed, const int generation){
	int num_new_mutations = 0;
	mutations.h_new_mutation_Indices[0] = mutations.h_mutations_Index;
	for(int pop = 0; pop < mutations.h_num_populations; pop++){
		int Nchrom_e = 2*demography(pop,generation)/(1+FI(pop,generation));
		if(Nchrom_e == 0 || mutations.extinct[pop]){ continue; }
		float mu = mu_rate(pop, generation);
		float lambda = mu*Nchrom_e*L;
		int temp = max(Rand1(lambda, lambda, mu, Nchrom_e*L, 1, generation, seed, pop),0);
		num_new_mutations += temp;
		mutations.h_new_mutation_Indices[pop+1] = num_new_mutations + mutations.h_mutations_Index;
	}
}

template <typename Functor_selection>
__host__ __forceinline__ void integrate_mse(double * d_mse_integral, const int N_ind, const int Nchrom_e, const Functor_selection sel_coeff, const float F, const float h, int pop, cudaStream_t pop_stream){
	double * d_freq;

	cudaMalloc((void**)&d_freq, Nchrom_e*sizeof(double));

	mse_integrand<Functor_selection> mse_fun(sel_coeff, N_ind, F, h, pop);
	trapezoidal_upper< mse_integrand<Functor_selection> > trap(mse_fun);

	calculate_area<<<10,1024,0,pop_stream>>>(d_freq, Nchrom_e, (double)1.0/(Nchrom_e), trap); //setup array frequency values to integrate over (upper integral from 1 to 0)

	void * d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_freq, d_mse_integral, Nchrom_e, pop_stream);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_freq, d_mse_integral, Nchrom_e, pop_stream);
	cudaFree(d_temp_storage);
	cudaFree(d_freq);

	reverse_array<<<10,1024,0,pop_stream>>>(d_mse_integral, Nchrom_e);
}

template <typename Functor_mutation, typename Functor_demography, typename Functor_selection, typename Functor_inbreeding>
__host__ __forceinline__ void initialize_mse(sim_struct & mutations, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_selection sel_coeff, const Functor_inbreeding FI, const float h, const int final_generation, const float num_sites, const int seed, const int compact_rate, cudaStream_t * pop_streams, cudaEvent_t * pop_events, const int myDevice){

	int num_freq = 0; //number of frequencies
	for(int pop = 0; pop < mutations.h_num_populations; pop++){
		float F = FI(pop,0);
		int Nchrom_e = 2*demography(pop,0)/(1+F);
		if(Nchrom_e <= 1){ continue; }
		num_freq += (int)(ceil((Nchrom_e - 1)/4.f)*4);
	} //adds a little padding to ensure distances between populations in array are a multiple of 4

	int * d_freq_index;
	cudaMalloc((void**)&d_freq_index, num_freq*sizeof(int));
	int * d_scan_index;
	cudaMalloc((void**)&d_scan_index, num_freq*sizeof(int));
	double ** mse_integral = new double *[mutations.h_num_populations];

	int offset = 0;
	for(int pop = 0; pop < mutations.h_num_populations; pop++){
		int N_ind = demography(pop,0);
		float mu = mu_rate(pop,0);
		float F = FI(pop,0);
		int Nchrom_e = 2*N_ind/(1+F);
		cudaMalloc((void**)&mse_integral[pop], Nchrom_e*sizeof(double));
		if(Nchrom_e <= 1){ continue; }
		integrate_mse(mse_integral[pop], N_ind, Nchrom_e, sel_coeff, F, h, pop, pop_streams[pop]);
		initialize_mse_frequency_array<<<6,1024,0,pop_streams[pop]>>>(d_freq_index, mse_integral[pop], offset, mu, N_ind, Nchrom_e, num_sites, sel_coeff, F, h, seed, pop);
		offset += (int)(ceil((Nchrom_e - 1)/4.f)*4);
	}

	for(int pop = 0; pop < mutations.h_num_populations; pop++){
		cudaFree(mse_integral[pop]);
		cudaEventRecord(pop_events[pop],pop_streams[pop]);
		cudaStreamWaitEvent(pop_streams[0],pop_events[pop],0);
	}

	delete [] mse_integral;

	void * d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_freq_index, d_scan_index, num_freq, pop_streams[0]);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_freq_index, d_scan_index, num_freq, pop_streams[0]);
	cudaFree(d_temp_storage);

	cudaEventRecord(pop_events[0],pop_streams[0]);
	for(int pop = 0; pop < mutations.h_num_populations; pop++){
		cudaStreamWaitEvent(pop_streams[pop],pop_events[0],0);
	}

	int prefix_sum_result;
	int final_freq_count;
	//final index is numfreq-1
	cudaMemcpy(&prefix_sum_result, &d_scan_index[(num_freq-1)], sizeof(int), cudaMemcpyDeviceToHost); //has to be in sync with host as result is used straight afterwards
	cudaMemcpy(&final_freq_count, &d_freq_index[(num_freq-1)], sizeof(int), cudaMemcpyDeviceToHost); //has to be in sync with host as result is used straight afterwards
	int num_mutations = prefix_sum_result+final_freq_count;
	set_Index_Length(mutations, num_mutations, mu_rate, demography, FI, num_sites, compact_rate, 0, final_generation);
	//cout<<"initial length " << mutations.h_array_Length << endl;

	cudaMalloc((void**)&mutations.d_mutations_freq, mutations.h_num_populations*mutations.h_array_Length*sizeof(float));
	cudaMalloc((void**)&mutations.d_prev_freq, mutations.h_num_populations*mutations.h_array_Length*sizeof(float));
	cudaMalloc((void**)&mutations.d_mutations_ID, mutations.h_array_Length*sizeof(int4));

	const dim3 blocksize(4,256,1);
	const dim3 gridsize(32,32,1);
	offset = 0;
	for(int pop = 0; pop < mutations.h_num_populations; pop++){
		float F = FI(pop,0);
		int Nchrom_e = 2*demography(pop,0)/(1+F);
		if(Nchrom_e <= 1){ continue; }
		initialize_mse_mutation_array<<<gridsize,blocksize,0,pop_streams[pop]>>>(mutations.d_prev_freq, d_freq_index, d_scan_index, offset, Nchrom_e, pop, mutations.h_num_populations, mutations.h_array_Length);
		offset += (int)(ceil((Nchrom_e - 1)/4.f)*4);
	}

	mse_set_mutID<<<50,1024,0,pop_streams[mutations.h_num_populations]>>>(mutations.d_mutations_ID, mutations.d_prev_freq, mutations.h_mutations_Index, mutations.h_num_populations, mutations.h_array_Length, myDevice);

	for(int pop = 0; pop <= mutations.h_num_populations; pop++){
		cudaEventRecord(pop_events[pop],pop_streams[pop]);
		for(int pop2 = mutations.h_num_populations+1; pop2 < 2*mutations.h_num_populations; pop2++){
			cudaStreamWaitEvent(pop_streams[pop2],pop_events[pop],0); //tells every pop_stream not used above to wait until initialization is done
		}
	}

	cudaFree(d_freq_index);
	cudaFree(d_scan_index);
}

//assumes prev_sim.num_sites is equivalent to current simulations num_sites or prev_sim.num_mutations == 0 (initialize to blank)
template <typename Functor_mutation, typename Functor_demography, typename Functor_inbreeding>
__host__ __forceinline__ void init_blank_prev_run(sim_struct & mutations, int & generation_shift, int & final_generation, const sim_result & prev_sim, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_inbreeding FI, const float num_sites, const int seed, const int compact_rate, cudaStream_t * pop_streams, cudaEvent_t * pop_events){
	//if prev_sim.num_mutations == 0 or num sites or num_populations between two runs are not equivalent, don't copy (initialize to blank)
	int num_mutations = 0;
	bool use_prev_sim = (num_sites == prev_sim.num_sites && mutations.h_num_populations == prev_sim.num_populations);
	if(use_prev_sim){
		generation_shift = prev_sim.total_generations;
		final_generation += generation_shift;
		num_mutations = prev_sim.num_mutations;
		mutations.extinct = new bool[mutations.h_num_populations];
		for(int i = 0; i < mutations.h_num_populations; i++){ mutations.extinct[i] = prev_sim.extinct[i]; }
	}

		set_Index_Length(mutations, num_mutations, mu_rate, demography, FI, num_sites, compact_rate, generation_shift, final_generation);
		cout<<"initial length " << mutations.h_array_Length << endl;
		cudaMalloc((void**)&mutations.d_mutations_freq, mutations.h_num_populations*mutations.h_array_Length*sizeof(float));
		cudaMalloc((void**)&mutations.d_prev_freq, mutations.h_num_populations*mutations.h_array_Length*sizeof(float));
		cudaMalloc((void**)&mutations.d_mutations_ID, mutations.h_array_Length*sizeof(int3));

	if(prev_sim.num_mutations != 0 && use_prev_sim){
		cudaMemcpy2DAsync(mutations.d_prev_freq, mutations.h_array_Length*sizeof(float), prev_sim.mutations_freq, prev_sim.num_mutations*sizeof(float), prev_sim.num_mutations*sizeof(float), prev_sim.num_populations, cudaMemcpyHostToDevice, pop_streams[0]);
		cudaMemcpyAsync(mutations.d_mutations_ID, prev_sim.mutations_ID, prev_sim.num_mutations*sizeof(int4), cudaMemcpyHostToDevice, pop_streams[1]);

		cudaEventRecord(pop_events[0],pop_streams[0]);
		cudaEventRecord(pop_events[1],pop_streams[1]);

		//wait until initialization is complete
		for(int pop = 2; pop < 2*mutations.h_num_populations; pop++){
			cudaStreamWaitEvent(pop_streams[pop],pop_events[0],0);
			cudaStreamWaitEvent(pop_streams[pop],pop_events[1],0);
		}
	}
}

template <typename Functor_mutation, typename Functor_demography, typename Functor_inbreeding>
__host__ __forceinline__ void compact(sim_struct & mutations, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_inbreeding FI, const float num_sites, const int generation, const int final_generation, const int compact_rate, cudaStream_t * control_streams, cudaEvent_t * control_events, cudaStream_t * pop_streams){
	int * d_flag;
	cudaMalloc((void**)&d_flag,mutations.h_mutations_Index*sizeof(int));

	flag_segregating_mutations<<<50,1024,0,control_streams[0]>>>(d_flag, mutations.d_prev_freq, mutations.h_num_populations, mutations.h_mutations_Index, mutations.h_array_Length);

	int * d_scan_Index;
	cudaMalloc((void**)&d_scan_Index, mutations.h_mutations_Index*sizeof(int));

	void * d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_flag, d_scan_Index, mutations.h_mutations_Index, control_streams[0]);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_flag, d_scan_Index, mutations.h_mutations_Index, control_streams[0]);
	cudaFree(d_temp_storage);

	int h_num_seg_mutations;
	cudaMemcpy(&h_num_seg_mutations, &d_scan_Index[mutations.h_mutations_Index-1], sizeof(int), cudaMemcpyDeviceToHost); //has to be in sync with the host since h_num_seq_mutations is manipulated on CPU right after
	h_num_seg_mutations += 1; //doesn't need to be h_num_seg_mutations += d_flag[mutations.h_mutations_Index-1] as last mutation in index will be new, so d_flag[mutations.h_mutations_Index-1] = 1

	int old_mutations_Index = mutations.h_mutations_Index;
	int old_array_Length = mutations.h_array_Length;
	set_Index_Length(mutations, h_num_seg_mutations, mu_rate, demography, FI, num_sites, compact_rate, generation, final_generation);

	float * d_temp;
	int4 * d_temp2;
	cudaMalloc((void**)&d_temp,mutations.h_num_populations*mutations.h_array_Length*sizeof(float));
	cudaMalloc((void**)&d_temp2,mutations.h_array_Length*sizeof(int4));

	const dim3 gridsize(50,mutations.h_num_populations,1);
	scatter_arrays<<<gridsize,1024,0,control_streams[0]>>>(d_temp, d_temp2, mutations.d_prev_freq, mutations.d_mutations_ID, d_flag, d_scan_Index, old_mutations_Index, mutations.h_array_Length, old_array_Length);
	cudaEventRecord(control_events[0],control_streams[0]);

	for(int pop = 0; pop < 2*mutations.h_num_populations; pop++){
		cudaStreamWaitEvent(pop_streams[pop],control_events[0],0);
	}

	cudaFree(mutations.d_prev_freq);
	cudaFree(mutations.d_mutations_ID);

	mutations.d_prev_freq = d_temp;
	mutations.d_mutations_ID = d_temp2;

	cudaFree(d_flag);
	cudaFree(d_scan_Index);

	cudaFree(mutations.d_mutations_freq);
	cudaMalloc((void**)&mutations.d_mutations_freq,mutations.h_num_populations*mutations.h_array_Length*sizeof(float));
}

__host__ __forceinline__ void swap_freq_pointers(sim_struct & mutations){
	float * temp = mutations.d_prev_freq;
	mutations.d_prev_freq = mutations.d_mutations_freq;
	mutations.d_mutations_freq = temp;
}

struct no_sample{
	__host__ __forceinline__ int operator()(const int generation) const{ return 0; }
};

template <typename Functor_mutation, typename Functor_demography, typename Functor_migration, typename Functor_selection, typename Functor_inbreeding, typename Functor_timesample = no_sample>
__host__ __forceinline__ sim_result * run_sim(const Functor_mutation mu_rate, const Functor_demography demography, const Functor_migration mig_prop, const Functor_selection sel_coeff, const Functor_inbreeding FI, const float h, const int num_generations, const float num_sites, const int num_populations, const int seed, Functor_timesample take_sample = Functor_timesample(), int max_samples = 0, const bool init_mse = true, const sim_result & prev_sim = sim_result(), const int compact_rate = 25, const int cuda_device = -1){
	int cudaDeviceCount;
	cudaGetDeviceCount(&cudaDeviceCount);
	if(cuda_device >= 0 && cuda_device < cudaDeviceCount){ cudaSetDevice(cuda_device); } //unless user specifies, driver auto-magically selects free GPU to run on
	int myDevice;
	cudaGetDevice(&myDevice);
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, myDevice);

	sim_struct mutations;
	sim_result * all_results = new sim_result[max_samples+1];

	mutations.h_num_populations = num_populations;
	mutations.h_new_mutation_Indices = new int[mutations.h_num_populations+1];
	mutations.extinct = new bool[mutations.h_num_populations];
	mutations.warp_size  = devProp.warpSize;

	cudaStream_t * pop_streams = new cudaStream_t[2*mutations.h_num_populations];
	cudaEvent_t * pop_events = new cudaEvent_t[2*mutations.h_num_populations];

	int num_control_streams = 4;
	cudaStream_t * control_streams = new cudaStream_t[num_control_streams];;
	cudaEvent_t * control_events = new cudaEvent_t[num_control_streams];;

	for(int pop = 0; pop < 2*mutations.h_num_populations; pop++){
		cudaStreamCreate(&pop_streams[pop]);
		cudaEventCreateWithFlags(&pop_events[pop],cudaEventDisableTiming);
	}

	for(int stream = 0; stream < num_control_streams; stream++){
		cudaStreamCreate(&control_streams[stream]);
		cudaEventCreateWithFlags(&control_events[stream],cudaEventDisableTiming);
	}

	int generation = 0;
	int final_generation = num_generations;
	//----- initialize simulation -----
	if(init_mse){
		//----- mutation-selection equilibrium (mse) (default) -----
		initialize_mse(mutations, mu_rate, demography, sel_coeff, FI, h, final_generation, num_sites, seed, compact_rate, pop_streams, pop_events, myDevice);
		//----- end -----
	}else{
		//----- initialize from results of previous simulation run or initialize to blank (blank will often take >> N generations to reach equilibrium) -----
		init_blank_prev_run(mutations, generation, final_generation, prev_sim, mu_rate, demography, FI, num_sites, seed, compact_rate, pop_streams, pop_events);
		//----- end -----
	}
	//----- end -----

	cout<< endl <<"initial num_mutations " << mutations.h_mutations_Index;

	//----- simulation steps -----

	int next_compact_generation = generation + compact_rate;
	int sample_index = 0;

	while((generation+1) <= final_generation){ //end of simulation
		generation++;

		//----- migration, selection, drift -----
		for(int pop = 0; pop < mutations.h_num_populations; pop++){
			int N_ind = demography(pop,generation);
			if(mutations.extinct[pop]){ continue; }
			if(N_ind <= 0){
				if(demography(pop,generation-1) > 0){ //previous generation, the population was alive
					N_ind = 0; //allow to go extinct
					mutations.extinct[pop] = true; //next generation will not process
				} else{ continue; } //if population has not yet arisen, it will have a population size of 0, can simply not process
			}
			float F = FI(pop,generation);
			int Nchrom_e = 2*N_ind/(1+F);
			//10^5 mutations: 600 blocks for 1 population, 300 blocks for 3 pops
			migration_selection_drift<<<600,128,0,pop_streams[pop]>>>(mutations.d_mutations_freq, mutations.d_prev_freq, mutations.h_mutations_Index, mutations.h_array_Length, Nchrom_e, mig_prop, sel_coeff, F, h, seed, pop, mutations.h_num_populations, generation);
		}
		//----- end -----

		//----- generate new mutations -----
		calc_new_mutations_Index(mutations, mu_rate, demography, FI, num_sites, seed, generation);
		for(int pop = 0; pop < mutations.h_num_populations; pop++){
			int N_ind = demography(pop,generation);
			if((N_ind <= 0) || mutations.extinct[pop]){ continue; }
			float F = FI(pop,generation);
			int Nchrom_e = 2*N_ind/(1+F);
			float freq = 1.f/Nchrom_e;
			int prev_Index = mutations.h_new_mutation_Indices[pop];
			int new_Index = mutations.h_new_mutation_Indices[pop+1];
			add_new_mutations<<<10,512,0,pop_streams[pop+mutations.h_num_populations]>>>(mutations.d_mutations_freq, mutations.d_mutations_ID, prev_Index, new_Index, mutations.h_array_Length, freq, pop, mutations.h_num_populations, generation, myDevice);
			cudaEventRecord(pop_events[pop+mutations.h_num_populations],pop_streams[pop+mutations.h_num_populations]);
		}
		mutations.h_mutations_Index = mutations.h_new_mutation_Indices[mutations.h_num_populations];
		//----- end -----

		for(int pop1 = 0; pop1 < mutations.h_num_populations; pop1++){
			for(int pop2 = 0; pop2 < mutations.h_num_populations; pop2++){
				cudaStreamWaitEvent(pop_streams[pop1],pop_events[pop2+mutations.h_num_populations],0); //wait to do the next round of mig_sel_drift until
			}
			if((take_sample(generation) && sample_index < max_samples && take_sample(generation) > 1) || (generation == next_compact_generation || generation == final_generation)){
				cudaStreamWaitEvent(control_streams[0],pop_events[pop1],0); //wait to compact/or record data until after mig_sel_drift and add_new_mut are done
				cudaStreamWaitEvent(control_streams[0],pop_events[pop1+mutations.h_num_populations],0);
			}
			if((take_sample(generation) && sample_index < max_samples)){
				cudaStreamWaitEvent(control_streams[1],pop_events[pop1],0);
				cudaStreamWaitEvent(control_streams[1],pop_events[pop1+mutations.h_num_populations],0);
				cudaStreamWaitEvent(control_streams[2],pop_events[pop1],0);
				cudaStreamWaitEvent(control_streams[2],pop_events[pop1+mutations.h_num_populations],0);
			}
		}
		if(cudaStreamQuery(control_streams[1]) != cudaSuccess){ cudaStreamSynchronize(control_streams[1]); } //if not yet done streaming to host from previous generation, pause here
		if(cudaStreamQuery(control_streams[2]) != cudaSuccess){ cudaStreamSynchronize(control_streams[2]); }

		swap_freq_pointers(mutations);

		//----- take time samples of frequency spectrum -----
		if(take_sample(generation) && sample_index < max_samples){
			//----- compact before sampling if requested -----
			if(take_sample(generation) > 1){ compact(mutations, mu_rate, demography, FI, num_sites, generation, final_generation, compact_rate, control_streams, control_events, pop_streams); next_compact_generation = generation + compact_rate; }
			//----- end -----
			sim_result::store_sim_result(all_results[sample_index], mutations, num_sites, generation, control_streams, control_events);
			sample_index++;
		}
		//----- end -----

		//----- compact every compact_rate generations and final generation -----
		if(generation == next_compact_generation || generation == final_generation){ compact(mutations, mu_rate, demography, FI, num_sites, generation, final_generation, compact_rate, control_streams, control_events, pop_streams); next_compact_generation = generation + compact_rate;}
		//----- end -----
	}
	//----- end -----

	//----- store final (compacted) generation on host -----
	sim_result::store_sim_result(all_results[max_samples], mutations, num_sites, generation, control_streams, control_events);
	//----- end -----

	cudaStreamSynchronize(control_streams[1]); //wait for writes to host to finish
	cudaStreamSynchronize(control_streams[2]);

	for(int pop = 0; pop < 2*mutations.h_num_populations; pop++){ cudaStreamDestroy(pop_streams[pop]); cudaEventDestroy(pop_events[pop]); }
	for(int stream = 0; stream < num_control_streams; stream++){ cudaStreamDestroy(control_streams[stream]); cudaEventDestroy(control_events[stream]); }

	delete pop_streams;
	delete pop_events;
	delete control_streams;
	delete control_events;

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
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float gamma = 0;
	float h = 0.5;
	float F = 1.0;
	int N_ind = pow(10.f,5)*(1+F); //constant population for now
	float s = gamma/(2*N_ind);
	float mu = pow(10.f,-9); //per-site mutation rate
	float L = 2*pow(10.f,7);
	float m = 0.01;
	int num_pop = 1;
	const int total_number_of_generations = pow(10.f,4);
	const int seed = 0xdecafbad;

	sim_result * a = run_sim(mutation(mu), demography(N_ind), mig_prop_pop(m,num_pop), sel_coeff(s), inbreeding(F), h, total_number_of_generations, L, num_pop, seed, no_sample(), 0, true);
	cout<<endl<<"final number of mutations: " << a[0].num_mutations << endl;
	delete [] a;

	//double * b = integrate_mse(N_ind, sel_coeff(s), F, h, 0, 0);
	//cudaFree(b);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("time elapsed: %f\n\n", elapsedTime);

	cudaDeviceReset();
}
