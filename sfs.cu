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

using namespace std;
using namespace cub;
using namespace r123;

__device__ int new_mutations_index;
__device__ int num_new_mutations;
__device__ int mutation_Index;
__device__ int array_length;

// uint_float_01: Input is a W-bit integer (unsigned).  It is multiplied
//	 by Float(2^-W) and added to Float(2^(-W-1)).  A good compiler should
//	 optimize it down to an int-to-float conversion followed by a multiply
//	 and an add, which might be fused, depending on the architecture.
//
//  If the input is a uniformly distributed integer, then the
//  result is a uniformly distributed floating point number in [0, 1].
//  The result is never exactly 0.0.
//  The smallest value returned is 2^-W.
//  Let M be the number of mantissa bits in Float.
//  If W>M  then the largest value retured is 1.0.
//  If W<=M then the largest value returned is the largest Float less than 1.0.
__device__ float uint_float_01(unsigned int in){
	//(mostly) stolen from Philox code "uniform.hpp"
	R123_CONSTEXPR float factor = float(1.)/(UINT_MAX + float(1.));
	R123_CONSTEXPR float halffactor = float(0.5)*factor;
    return in*factor + halffactor;
}


// uint_float_neg11: Input is a W-bit integer (unsigned).  It is cast
//    to a W-bit signed integer, multiplied by Float(2^-(W-1)) and
//    then added to Float(2^(-W-2)).  A good compiler should optimize
//    it down to an int-to-float conversion followed by a multiply and
//    an add, which might be fused, depending on the architecture.
//
//  If the input is a uniformly distributed integer, then the
//  output is a uniformly distributed floating point number in [-1, 1].
//  The result is never exactly 0.0.
//  The smallest absolute value returned is 2^-(W-1)
//  Let M be the number of mantissa bits in Float.
//  If W>M  then the largest value returned is 1.0 and the smallest is -1.0.
//  If W<=M then the largest value returned is the largest Float less than 1.0
//    and the smallest value returned is the smallest Float greater than -1.0.
__device__ float uint_float_neg11(unsigned int in){
	//(mostly) stolen from Philox code "uniform.hpp"
	R123_CONSTEXPR float factor = float(1.)/(INT_MAX + float(1.));
	R123_CONSTEXPR float halffactor = float(0.5)*factor;
    return int(in)*factor + halffactor;
}

__device__ float2 boxmuller(const unsigned int u0, const unsigned int u1, const float mean1, const float mean2, const float var1, const float var2) {
	//(mostly) stolen from Philox code "boxmuller.hpp"
    float r1, r2;
    float2 f;
    sincospif(uint_float_neg11(u0), &f.x, &f.y); //-1_1 is necessary because this is sin(pi*x), cos(pi*x) not sin(2pi*x) and cos(2pi*x) - could've also clamped it between 0 and 1 and multplied x by 2
    float temp = -2.f * logf(uint_float_01(u1));
    r1 = sqrtf(var1 * temp);
    r2 = sqrtf(var2 * temp);
    f.x = f.x*r1 + mean1;
    f.y = f.y*r2 + mean2;
    return f;
}

__device__ int4 round(float4 f){ return make_int4(round(f.x), round(f.y), round(f.z), round(f.w)); }

__device__ int RandNorm1(float mean, float var, int k, int step, int seed, int population){
	//Normal approximation to Binomial (mean = Np, std = sqrt(var = Np(1-p))) and Poisson (mean = Np, std = sqrt(var = mean))

	typedef Philox4x32_R<8> P; //can change the 10 rounds of bijection down to 8 (lowest safe limit) to get possible extra speed!
    P rng;

	P::key_type key = {{k, seed}};
    P::ctr_type count = {{step, population, 0xdeadbeef, 0xbeeff00d}}; //random ints to set counter space

	union {
	    P::ctr_type c;
	    uint4 i;
	}u;

	u.c = rng(count, key);

	float2 G = boxmuller(u.i.x,u.i.y,mean,mean,var,var);

	return round(G.x);
}

__device__ int4 RandNorm4(float4 mean, float4 var, int k, int step, int seed, int population){
	//Normal approximation to Binomial (mean = Np, std = sqrt(var = Np(1-p))) and Poisson (mean = Np, std = sqrt(var = mean))

	typedef Philox4x32_R<8> P; //can change the 10 rounds of bijection down to 8 (lowest safe limit) to get possible extra speed!
    P rng;

	P::key_type key = {{k, seed}};
    P::ctr_type count = {{step, population, 0xdeadbeef, 0xbeeff00d}}; //random ints to set counter space

	union {
	    P::ctr_type c;
	    uint4 i;
	}u;

	u.c = rng(count, key);

	float2 G1 = boxmuller(u.i.x,u.i.y,mean.x,mean.y,var.x,var.y);
	float2 G2 = boxmuller(u.i.z,u.i.w,mean.z,mean.w,var.z,var.w);

	return round(make_float4(G1.x,G1.y,G2.x,G2.y));
}

__device__ float4 exp(float4 f){ return(make_float4(exp(f.x),exp(f.y),exp(f.z),exp(f.w))); }

__global__ void initialize_frequency_array(int * const freq_index, const float mu, const int N, const int L, const float s, const float h, const int seed, const int population){
	//determines number of mutations at each frequency in the initial population, sets it equal to mutation-selection balance
	int myID = blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < (N-1)/4; id+= blockDim.x*gridDim.x){ //exclusive, length of freq array is chromosome population size N-1
		float4 i = make_float4((4*id + 1),(4*id + 2),(4*id + 3),(4*id + 4))/float(N);
		float4 lambda;
		if(s == 0){ lambda = 2*mu*L/i; }
		else{ lambda =  2*mu*L*(-1*exp(-1*(2*N*s)*(-1.0*i+1.0))+1.0)/((-1*exp(-1*(2*N*s))+1)*i*(-1.0*i+1.0)); }
		reinterpret_cast<int4*>(freq_index)[id] = clamp(RandNorm4(lambda, lambda, 0, id, seed, population),0,N*L); //round(lambda);//// ////mutations are poisson distributed in each frequency class
		//printf("%d %d %f %f %f %f %f %f %f %f \r", myID, id, i.x, i.y, i.z, i.w, lambda.x, lambda.y, lambda.z, lambda.w);
	}


	int id = myID + (N-1)/4*4; //all integers //right now only works if minimum of 3 threads are launched
	if(id < (N-1)){
		float i = float(id+1)/float(N);
		float lambda;
		if(s == 0){ lambda = 2*mu*L/i; }
		else{ lambda =  2*mu*L*(1-exp(-1*(2*N*s)*(1-i)))/((1-exp(-1*(2*N*s)))*i*(1-i)); }
		freq_index[id] = clamp(RandNorm1(lambda, lambda, 0, id, seed, population),0,N*L);//round(lambda);// //  //mutations are poisson distributed in each frequency class
		//printf("%d %d %f %f\r", myID, id, i, lambda);
	}
}

__global__ void set_Index_Length(const int * scan_index, const int * freq_index, const int N){
	//one thread only, final index in N-2 (N-1 terms)
	mutation_Index = scan_index[(N-2)]+freq_index[(N-2)]-1; //mutation_Index equal to num_mutations-1 (zero-based indexing) at initialization
	array_length = mutation_Index + 10*sqrtf(mutation_Index);	//apparently due tails truncation can't get more than 6.66 stdevs from the mean with Box-Muller (http://en.wikipedia.org/wiki/Box�Muller_transform), of course multiple generations ...
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

__global__ void selection_drift(float * mutations, int * is_zero, const int N, const float s, const float h, const int seed, const int population, const int counter, const int generation){
	//calculates new frequencies for every mutation in the population, N1 previous pop size, N2, new pop size
	//myID+seed for random number generator philox's k, generation+pop_offset for its step in the pseudorandom sequence
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < (mutation_Index+1)/4; id+= blockDim.x*gridDim.x){ //inclusive for mutation_Index
		float4 i = reinterpret_cast<float4*>(mutations)[id]; //allele frequency in previous population size
		float4 p = (1+s)*i/((1+s)*i + 1*(-1*i + 1.0)); //haploid
		//p = ((1+s)*i*i+(1+h*s)*i*(1-i))/((1+s)*i*i + 2*(1+h*s)*i*(1-i) + (1-i)*(1-i)); //diploid
		float4 mean = p*float(N); //expected allele frequency in new generation's population size
		int4 j = clamp(RandNorm4(mean,(-1*p + 1.0)*mean,(id + 2),counter,seed,population),0, N);//round(mean);//
		int4 k = make_int4(0);
		if(j.x == N || j.x == 0){ j.x = 0; k.x = 1; /*if(j.x==N) { atomicAdd(fixed_diff_count, 1); }*/ } //for multiple populations only do this if derived allele has gone to fixation or loss in all populations
		if(j.y == N || j.y == 0){ j.y = 0; k.y = 1; /*if(j.y==N) { atomicAdd(fixed_diff_count, 1); }*/ }
		if(j.z == N || j.z == 0){ j.z = 0; k.z = 1; /*if(j.z==N) { atomicAdd(fixed_diff_count, 1); }*/ }
		if(j.w == N || j.w == 0){ j.w = 0; k.w = 1; /*if(j.w==N) { atomicAdd(fixed_diff_count, 1); }*/ }
		reinterpret_cast<float4*>(mutations)[id] = make_float4(j)/float(N); //final allele freq
		reinterpret_cast<int4*>(is_zero)[id] = k;
	}
	int id = myID + (mutation_Index+1)/4 * 4;  //right now only works if minimum of 3 threads are launched
	if(id <= mutation_Index){ //inclusive for mutation_Index
		float i = mutations[id]; //allele frequency in previous population size
		float p = (1+s)*i/((1+s)*i + 1*(1.0-i)); //haploid
		//p = ((1+s)*i*i+(1+h*s)*i*(1-i))/((1+s)*i*i + 2*(1+h*s)*i*(1-i) + (1-i)*(1-i)); //diploid
		float mean = p*float(N); //expected allele frequency in new generation's population size
		int j = clamp(RandNorm1(mean,(1.0-p)*mean,(id + 2),counter,seed,population),0, N); //round(mean);//
		int k = 0;
		if(j == N || j == 0){ j = 0; k = 1; /*if(j==N) { atomicAdd(fixed_diff_count, 1); }*/ } //for multiple populations only do this if derived allele has gone to fixation or loss in all populations
		mutations[id] = float(j)/float(N); //final allele freq
		is_zero[id] = k;
	}
}

__global__ void search_new_mutations_index(int * is_zero, int * is_zero_inclusive_scan){
	if(new_mutations_index == -1){
		int myID =  blockIdx.x*blockDim.x + threadIdx.x;

		for(int id = myID; id < mutation_Index/4; id+= blockDim.x*gridDim.x){ //exclusive for mutation_Index
			int4 i = reinterpret_cast<int4*>(is_zero_inclusive_scan)[id];
			int4 j = reinterpret_cast<int4*>(is_zero)[id];
			if(i.x == num_new_mutations && j.x == 1){ new_mutations_index = 4*id; } //can only be one answer
			if(i.y == num_new_mutations && j.y == 1){ new_mutations_index = 4*id+1; }
			if(i.z == num_new_mutations && j.z == 1){ new_mutations_index = 4*id+2; }
			if(i.w == num_new_mutations && j.w == 1){ new_mutations_index = 4*id+3; }
		}
		int id = myID + (mutation_Index)/4 * 4;  //right now only works if minimum of 3 threads are launched
		if(id < mutation_Index){ //exclusive for mutation_Index
			int i = is_zero_inclusive_scan[id];
			int j = is_zero[id];
			if(i == num_new_mutations && j == 1){ new_mutations_index = id; }
		}
	}
}

__global__ void calc_new_mutations_index(int * is_zero_inclusive_scan, const int mu, const int N, const int L, const int seed, const int population, const int counter){
	//run with 1 thread
	float lambda = mu*N*L;
	num_new_mutations = clamp(RandNorm1(lambda,lambda,1,counter,seed,population),0, N*L);
	int total_zeros = is_zero_inclusive_scan[mutation_Index];
	new_mutations_index = -1;
	if(num_new_mutations >= total_zeros){
		//for now if num_new_mutations_index > last index, make it equal to last index, do not resize array
		mutation_Index = min(((num_new_mutations-total_zeros)+mutation_Index), array_length-1);
		new_mutations_index = mutation_Index;
	}
}

__global__ void add_new_mutations(float * mutations, float allele_freq){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < (new_mutations_index+1)/4; id+= blockDim.x*gridDim.x){ //inclusive for num_new_mutations_index
		float4 i = reinterpret_cast<float4*>(mutations)[id];
		if(i.x == 0){ i.x = allele_freq;}
		if(i.y == 0){ i.y = allele_freq;}
		if(i.z == 0){ i.z = allele_freq;}
		if(i.w == 0){ i.w = allele_freq;}
		reinterpret_cast<float4*>(mutations)[id] = i;
	}
	int id = myID + (new_mutations_index+1)/4 * 4;  //right now only works if minimum of 3 threads are launched
	if(id <= new_mutations_index){ //inclusive for num_new_mutations_index
		if(mutations[id] == 0){ mutations[id] = allele_freq; }
	}
}


__host__ __forceinline__ void new_mutations(float * mutations, int * is_zero, int * is_zero_inclusive_scan, const int mu, const int N, const int L, const int seed, const int population, const int counter){
	//adds new mutations to float population
	//by replacing all mutations up to and equal to num_new_mutation with 0 allele count with 1

	//mutation_index is the highest index in the array reached by the simulation
	//it is only equal to num_new_mutation_index if num_new_mutations > mutation_index
	//calculate num_new_mutation_index

	//potential speed up? : serially_scan_over_blocks(cub block reduce is_zero over threads <= mutation_Index) until num_zeros >= num_new_mutations
	//if num_zeros < num_new_mutations, then mutation_Index  += num_new_mutations - num_zeros; add_mutations<<>>(mutations,mutation_Index);
	//else cub_scan_block where num_zeros >= num_new_mutations; determine thread_ID where num_zeros = num_new_mutations; add_mutations<<>>(mutations,myID);
	int h_mutation_Index;
	cudaMemcpyFromSymbol(&h_mutation_Index, mutation_Index, sizeof(array_length), 0, cudaMemcpyDeviceToHost);
	//cout<< endl << h_mutation_Index << endl;
	void * d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CubDebugExit(DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, is_zero, is_zero_inclusive_scan, (h_mutation_Index+1)));
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    CubDebugExit(DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, is_zero, is_zero_inclusive_scan, (h_mutation_Index+1)));
	cudaFree(d_temp_storage);

	calc_new_mutations_index<<<1,1>>>(is_zero_inclusive_scan, mu, N, L, seed, population, counter); //only run with one thread
	search_new_mutations_index<<<16, 1024>>>(is_zero, is_zero_inclusive_scan);
	add_new_mutations<<<16, 1024>>>(mutations,1.f/float(N));
}

__host__ __forceinline__ void one_generation(float * mutations, int * is_zero, int * is_zero_inclusive_scan, const float mu, const int N, const int L, const float s, const int h, const int seed, const int population, const int counter, const int generation){
	selection_drift<<<1000, 64 >>>(mutations, is_zero, N, s, h, seed, population, counter, generation);
	new_mutations(mutations, is_zero, is_zero_inclusive_scan, mu, N, L, seed, population, counter);
	//migration amongst the new generation
}

__host__ __forceinline__ void run_sim(const float mu, const int N, const float s, const int h, const int L, const int total_number_of_generations, const int burn_in, const int seed){
	int * is_zero;
	int * is_zero_inclusive_scan;
	float * mutations; //allele counts of all current mutations
	int population = 0;
	//initialize_equilibrium(mutations, is_zero, is_zero_inclusive_scan, mu, N, L, s, h, seed, 0);

	int num_bytes = (N-1)*sizeof(int);
	int * freq_index;
	cudaMalloc((void**)&freq_index, num_bytes);
	int * scan_index;
	cudaMalloc((void**)&scan_index,num_bytes);

	initialize_frequency_array<<<6, 1024>>>(freq_index, mu, N, L, s, h, seed, population);
	//cudaDeviceSynchronize();
	//print_Device_array_int<<<1,1>>>(freq_index,(N-1));
	//sum_Device_array_int<<<1,1>>>(freq_index,(N-1));
	void * d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, freq_index, scan_index, (N-1));
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, freq_index, scan_index, (N-1));
	cudaFree(d_temp_storage);
	cout<<endl;
	//print_Device_array_int<<<1,1>>>(scan_index,(N-1));

	set_Index_Length<<<1,1>>>(scan_index, freq_index, N);

	int h_array_length;
	cudaDeviceSynchronize();
	cudaMemcpyFromSymbol(&h_array_length, array_length, sizeof(array_length), 0, cudaMemcpyDeviceToHost);
	cout<<endl<<"length " << h_array_length << endl;
	num_bytes = h_array_length*sizeof(float);
	cudaMalloc((void**)&mutations, num_bytes);
	num_bytes = h_array_length*sizeof(int);
	cudaMalloc((void**)&is_zero, num_bytes);
	cudaMalloc((void**)&is_zero_inclusive_scan, num_bytes);
	const dim3 blocksize(4,256,1);
	const dim3 gridsize(16,32,1);
	initialize_mutation_array<<<gridsize, blocksize>>>(mutations, freq_index,scan_index, N);

	//print_Device_array_int<<<1,1>>>(mutations,h_array_length);

	cudaFree(freq_index);
	cudaFree(scan_index);

	//print_Device_array_int<<<1,1>>>(mutations,50);
	cout<<endl;
	#pragma unroll
	for(int counter = 0; counter < burn_in; counter++){ one_generation(mutations, is_zero, is_zero_inclusive_scan, mu, N, L, s, h, seed, 0, counter, 0); }
	cudaEvent_t start, stop;
	    cudaEventCreate(&start);
	    cudaEventCreate(&stop);
	    cudaEventRecord(start, 0);
	#pragma unroll
	for(int counter = burn_in; counter < (burn_in+total_number_of_generations); counter++){ one_generation(mutations, is_zero, is_zero_inclusive_scan, mu, N, L, s, h, seed, 0, counter, (counter-burn_in+1)); }
	cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);
		printf("time elapsed generations: %f\n", elapsedTime);
	cudaFree(mutations);
	cudaFree(is_zero);
	cudaFree(is_zero_inclusive_scan);
}

int main(int argc, char **argv)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	/*int max_sm_occupancy;
    CubDebugExit(MaxSmOccupancy(max_sm_occupancy, BlockSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM>, BLOCK_THREADS));*/ //part of CUB ... could be useful ...

	int N_chrom_pop = 2*pow(10.f,4); //constant population for now
	float s = 0; //neutral for now
	float h = 0.5;
	float mu = pow(10.f,-9); //per-site mutation rate
	int L = 2.5*pow(10.f,8); //eventually set so so the number of expected mutations is > a certain amount
	//int N_chrom_samp = 200;
	const int total_number_of_generations = pow(10.f,4);
	const int burn_in = 0;
	const int seed = 0xdecafbad;


	run_sim(mu, N_chrom_pop, s, h, L, total_number_of_generations, burn_in, seed);
	//cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("time elapsed: %f\n", elapsedTime);
}
