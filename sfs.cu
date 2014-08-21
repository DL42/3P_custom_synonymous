#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <limits>
#include<sys/time.h>

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

__device__ int mutations_Index; //number of mutations in the population(s)
__device__ int array_length;
__device__ int new_mutations_Index; //number of mutations in the population(s) in the new generation (after new mutations enter population(s))

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
__device__ float uint_float_01(unsigned int in){
	//(mostly) stolen from Philox code "uniform.hpp"
	R123_CONSTEXPR float factor = float(1.)/(UINT_MAX + float(1.));
	R123_CONSTEXPR float halffactor = float(0.5)*factor;
    return in*factor + halffactor;
}

__device__ int4 round(float4 f){ return make_int4(round(f.x), round(f.y), round(f.z), round(f.w)); }

__device__ float4 exp(float4 f){ return(make_float4(exp(f.x),exp(f.y),exp(f.z),exp(f.w))); }

__device__ uint4 Philox(int k, int step, int seed, int population, int round){
	typedef Philox4x32_R<8> P; //can change the 10 rounds of bijection down to 8 (lowest safe limit) to get possible extra speed!
	P rng;

	P::key_type key = {{k, seed}};
	P::ctr_type count = {{step, population, round, 0xbeeff00d}}; //random ints to set counter space

	union {
		P::ctr_type c;
		uint4 i;
	}u;

	u.c = rng(count, key);

	return u.i;
}

__device__ int poiscdfinv(float p, float mean){
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

__device__ int RandBinom(float p, float N, int k, int step, int seed, int population, int start_round){
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

__device__ int Rand1(float mean, float var, float p, float N, int k, int step, int seed, int population){

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

__global__ void initialize_frequency_array(int * const freq_index, const float mu, const int N, const int L, const float s, const float h, const int seed, const int population){
	//determines number of mutations at each frequency in the initial population, sets it equal to mutation-selection balance
	int myID = blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < (N-1)/4; id+= blockDim.x*gridDim.x){ //exclusive, length of freq array is chromosome population size N-1
		float4 i = make_float4((4*id + 1),(4*id + 2),(4*id + 3),(4*id + 4))/float(N);
		float4 lambda;
		if(s == 0){ lambda = 2*mu*L/i; }
		else{ lambda =  2*mu*L*(-1*exp(-1*(2*N*s)*(-1.0*i+1.0))+1.0)/((-1*exp(-1*(2*N*s))+1)*i*(-1.0*i+1.0)); }
		reinterpret_cast<int4*>(freq_index)[id] = max(Rand4(lambda, lambda, make_float4(mu), L*float(N), 0, id, seed, population),make_int4(0)); //round(lambda);//// ////mutations are poisson distributed in each frequency class
		//printf("%d %d %f %f %f %f %f %f %f %f \r", myID, id, i.x, i.y, i.z, i.w, lambda.x, lambda.y, lambda.z, lambda.w);
	}


	int id = myID + (N-1)/4*4; //all integers //right now only works if minimum of 3 threads are launched
	if(id < (N-1)){
		float i = float(id+1)/float(N);
		float lambda;
		if(s == 0){ lambda = 2*mu*L/i; }
		else{ lambda =  2*mu*L*(1-exp(-1*(2*N*s)*(1-i)))/((1-exp(-1*(2*N*s)))*i*(1-i)); }
		freq_index[id] = max(Rand1(lambda, lambda, mu, L*float(N), 0, id, seed, population),0);//round(lambda);// //  //mutations are poisson distributed in each frequency class
		//printf("%d %d %f %f\r", myID, id, i, lambda);
	}
}

__global__ void set_Index_Length(const int * scan_index, const int * freq_index, const float mu, const int N, const int L, const int compact){
	//one thread only, final index in N-2 (N-1 terms)
	mutations_Index = scan_index[(N-2)]+freq_index[(N-2)]-1; //mutation_Index equal to num_mutations-1 (zero-based indexing) at initialization
	array_length = mutations_Index + (mu*N*L + 7*sqrtf(mu*N*L))*compact;
	//printf("\r %d %d \r",mutation_Index,array_length);
}

__global__ void initialize_mutation_array(float * mutations, int * freq_index, int * scan_index, const int N){
	//fills in mutation array using the freq and scan indices
	//y threads correspond to freq_index/scan_index indices, use grid-stride loops
	//x threads correspond to mutation array indices, use grid-stride loops
	//using scan number to define start of array, freq_index to define num_new_mutations_index (if 0 simply ignore) and myIDx used to calculate allele_count
	int myIDy = blockIdx.y*blockDim.y + threadIdx.y;
	for(int idy = myIDy; idy < (N-1); idy+= blockDim.y*gridDim.y){
		int myIDx = blockIdx.x*blockDim.x + threadIdx.x;
		int start = scan_index[idy];
		int num_mutations = freq_index[idy];
		float freq = float(idy+1)/float(N);
		for(int idx = myIDx; idx < num_mutations; idx+= blockDim.x*gridDim.x){ mutations[start + idx] = freq; }
	}
}

/*__global__ void print_Device_array_int(float * array, int num){

	for(int i = 9500; i < 10500; i++){
		if(i%1000 == 0){ printf("\r"); }
		printf("%f ",array[i]);
	}
}

__global__ void sum_Device_array_int(int * array, int num){
	int j = 0;
	for(int i = 0; i < num; i++){
		j += array[i];
	}
	printf("\r%d\r",j);
}*/

__global__ void selection_drift(float * mutations, const int N, const float s, const float h, const int seed, int population, const int counter, const int generation){
	//calculates new frequencies for every mutation in the population, N1 previous pop size, N2, new pop size
	//myID+seed for random number generator philox's k, generation+pop_offset for its step in the pseudorandom sequence
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < mutations_Index/4; id+= blockDim.x*gridDim.x){
		float4 i = reinterpret_cast<float4*>(mutations)[id]; //allele frequency in previous population size
		float4 p = (1+s)*i/((1+s)*i + 1*(-1*i + 1.0)); //haploid
		//p = ((1+s)*i*i+(1+h*s)*i*(1-i))/((1+s)*i*i + 2*(1+h*s)*i*(1-i) + (1-i)*(1-i)); //diploid
		float4 mean = p*float(N); //expected allele frequency in new generation's population size
		int4 j = clamp(Rand4(mean,(-1*p + 1.0)*mean,p, N,(id + 2),counter,seed,population),0, N);
		reinterpret_cast<float4*>(mutations)[id] = make_float4(j)/float(N); //final allele freq
	}
	int id = myID + mutations_Index/4 * 4;  //right now only works if minimum of 3 threads are launched
	if(id < mutations_Index){
		float i = mutations[id]; //allele frequency in previous population size
		float p = (1+s)*i/((1+s)*i + 1*(1.0-i)); //haploid
		//p = ((1+s)*i*i+(1+h*s)*i*(1-i))/((1+s)*i*i + 2*(1+h*s)*i*(1-i) + (1-i)*(1-i)); //diploid
		float mean = p*float(N); //expected allele frequency in new generation's population size
		int j = clamp(Rand1(mean,(1.0-p)*mean,p,N,(id + 2),counter,seed,population),0, N); //round(mean);//
		mutations[id] = float(j)/float(N); //final allele freq
	}
}

__global__ void num_new_mutations(const float mu, const int N, const int L, const int seed, const int counter, const int generation){
	//1 thread 1 block for now
	int lambda = mu*N*L;
	int num_new_mutations = max(Rand1(lambda, lambda, mu, float(N)*L, 1, counter, seed, 0),0);
	new_mutations_Index = num_new_mutations + mutations_Index;
}

__global__ void copy_array(float * smaller_array, float * larger_array){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < mutations_Index/4; id+= blockDim.x*gridDim.x){
		reinterpret_cast<float4*>(larger_array)[id] = reinterpret_cast<float4*>(smaller_array)[id];
	}
	int id = myID + mutations_Index/4 * 4;  //right now only works if minimum of 3 threads are launched
	if(id < mutations_Index){ larger_array[id] = smaller_array[id]; }
}

__global__ void add_new_mutations(float * new_mutations, float freq){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; (id < (new_mutations_Index-mutations_Index)) && ((id + mutations_Index) < array_length); id+= blockDim.x*gridDim.x){ new_mutations[(mutations_Index+id)] = freq; }
}

__global__ void reset_index(){
	//run with 1 thread 1 block
	mutations_Index = new_mutations_Index;
}

__global__ void set_length(int * Length, const float mu, const int N, const int L, const int compact){
	//run with 1 thread 1 block
	mutations_Index = Length[0];
	array_length = mutations_Index + (mu*N*L + 7*sqrtf(mu*N*L))*compact;
}

struct Clamp
{
	__host__ __device__ __forceinline__ bool operator()(const float &a) const {
        return (a > 0.f && a < 1.f);
    }
};

__host__ __forceinline__ void run_sim(const float mu, const int N, const float s, const int h, const int L, const int total_sim_generations, const int burn_in, const int seed){

	float * mutations; //allele counts of all current mutations
	int population = 0;

	//----- initialize simulation -----
	int num_bytes = (N-1)*sizeof(int);
	int * freq_index;
	cudaMalloc((void**)&freq_index, num_bytes);
	int * scan_index;
	cudaMalloc((void**)&scan_index,num_bytes);
	int compact = 40;
	initialize_frequency_array<<<6, 1024>>>(freq_index, mu, N, L, s, h, seed, population);

	void * d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, freq_index, scan_index, (N-1));
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, freq_index, scan_index, (N-1));
	cudaFree(d_temp_storage);

	set_Index_Length<<<1,1>>>(scan_index, freq_index, mu, N, L,compact);

	int h_array_length;
	cudaMemcpyFromSymbol(&h_array_length, array_length, sizeof(array_length), 0, cudaMemcpyDeviceToHost);
	cout<<endl<<"initial length " << h_array_length << endl;
	num_bytes = h_array_length*sizeof(float);
	cudaMalloc((void**)&mutations, num_bytes);

	const dim3 blocksize(4,256,1);
	const dim3 gridsize(16,32,1);
	initialize_mutation_array<<<gridsize, blocksize>>>(mutations, freq_index,scan_index, N);

	cudaFree(freq_index);
	cudaFree(scan_index);
	//----- end -----

	//----- burn in steps -----
    #pragma unroll
	for(int counter = 0; counter < burn_in; counter++){

		selection_drift<<<1000,64>>>(mutations, N, s, h, seed, population, counter, 0);

		//----- generate new mutations -----
		num_new_mutations<<<1,1>>>(mu, N, L, seed, counter, 0);
		add_new_mutations<<<5,1024>>>(mutations,1.f/float(N));
		reset_index<<<1,1>>>();
		//----- end -----

		//----- compact every X generations -----
		if((counter > 0 && counter % compact == 0) || counter == (burn_in - 1)){
			float * temp = NULL;
			int * length = NULL;
			cudaMalloc((void**)&temp,num_bytes);
			cudaMalloc((void**)&length,sizeof(int));
			d_temp_storage = NULL;
			size_t temp_storage_bytes = 0;
			Clamp select_op;
			cudaMemcpyFromSymbol(&h_array_length, mutations_Index, sizeof(mutations_Index), 0, cudaMemcpyDeviceToHost);
			cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, mutations, temp, length, h_array_length, select_op);
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
			cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, mutations, temp, length, h_array_length, select_op);
			cudaFree(d_temp_storage);

			set_length<<<1,1>>>(length, mu, N, L, compact);
			cudaMemcpyFromSymbol(&h_array_length, array_length, sizeof(array_length), 0, cudaMemcpyDeviceToHost);
			float * temp2;
			num_bytes = h_array_length*sizeof(float);
			cudaMalloc((void**)&temp2, num_bytes);
			copy_array<<<50,1024>>>(temp, temp2);
			cudaFree(temp);
			cudaFree(mutations);
			mutations = temp2;
		}
		//----- end -----
	}
	//----- end -----
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//----- simulation steps -----
	int generations;
    #pragma unroll
	for(int counter = burn_in; counter < (burn_in+total_sim_generations); counter++){
		generations = counter - burn_in + 1;
		selection_drift<<<1000, 64 >>>(mutations, N, s, h, seed, population, counter, generations);

		//-----generate new mutations -----

		num_new_mutations<<<1,1>>>(mu, N, L, seed, counter, 0);
		add_new_mutations<<<5,1024>>>(mutations,1.f/float(N));
		reset_index<<<1,1>>>();
		//----- end -----

		//-----compact every X generations -----
		if((generations % compact == 0) || counter == (burn_in+total_sim_generations - 1)){
			float * temp = NULL;
			int * length = NULL;
			cudaMalloc((void**)&temp,num_bytes);
			cudaMalloc((void**)&length,sizeof(int));
			d_temp_storage = NULL;
			size_t temp_storage_bytes = 0;
			Clamp select_op;
			cudaMemcpyFromSymbol(&h_array_length, mutations_Index, sizeof(mutations_Index), 0, cudaMemcpyDeviceToHost);
			cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, mutations, temp, length, h_array_length, select_op);
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
			cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, mutations, temp, length, h_array_length, select_op);
			cudaFree(d_temp_storage);

			set_length<<<1,1>>>(length, mu, N, L, compact);
			cudaMemcpyFromSymbol(&h_array_length, array_length, sizeof(array_length), 0, cudaMemcpyDeviceToHost);
			float * temp2;
			num_bytes = h_array_length*sizeof(float);
			cudaMalloc((void**)&temp2, num_bytes);
			copy_array<<<50,1024>>>(temp, temp2);
			cudaFree(temp);
			cudaFree(mutations);
			mutations = temp2;
			//int out;
			//cudaMemcpyFromSymbol(&out, mutations_Index, sizeof(mutations_Index), 0, cudaMemcpyDeviceToHost);
			//cout<<endl<<"number of mutations: " << out << endl;
		}
		//----- end -----

	}
	//----- end -----
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("time elapsed generations: %f\n", elapsedTime);
	int out;
	cudaMemcpyFromSymbol(&out, mutations_Index, sizeof(mutations_Index), 0, cudaMemcpyDeviceToHost);
	cout<<endl<<"final number of mutations: " << out << endl;
	cudaFree(mutations);
}

int main(int argc, char **argv)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	int N_chrom_pop = 2*pow(10.f,4); //constant population for now
	float s = 0; //neutral for now
	float h = 0.5;
	float mu = pow(10.f,-9); //per-site mutation rate
	int L = 2.5*pow(10.f,8); //eventually set so so the number of expected mutations is > a certain amount, although L < 2^32 (max of int)
	//int N_chrom_samp = 200;
	const int total_number_of_generations = pow(10.f,4);
	const int burn_in = 0;
	const int seed = 0xdecafbad;

	run_sim(mu, N_chrom_pop, s, h, L, total_number_of_generations, burn_in, seed);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("time elapsed: %f\n", elapsedTime);
}
