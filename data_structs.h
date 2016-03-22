/*
 * data_structs.h
 *
 *  Created on: Mar 20, 2016
 *      Author: dlawrie
 */

#ifndef DATA_STRUCTS_H_
#define DATA_STRUCTS_H_

#include <stdio.h>

/* ----- error checking ----- */
#define __DEBUG__ false
#define cudaCheckErrors(expr1,expr2,expr3) { cudaError_t e = expr1; int g = expr2; int p = expr3; if (e != cudaSuccess) { fprintf(stderr,"error %d %s\tfile %s\tline %d\tgeneration %d\t population %d\n", e, cudaGetErrorString(e),__FILE__,__LINE__, g,p); exit(1); } }
#define cudaCheckErrorsAsync(expr1,expr2,expr3) { cudaCheckErrors(expr1,expr2,expr3); if(__DEBUG__){ cudaCheckErrors(cudaDeviceSynchronize(),expr2,expr3); } }
/* ----- end of error checking ----- */

//for internal simulation function passing
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

	sim_struct();

	~sim_struct();
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

	sim_result();
	void static store_sim_result(sim_result & out, sim_struct & mutations, int num_sites, int total_generations, cudaStream_t * control_streams, cudaEvent_t * control_events);
	~sim_result();
};

struct sfs{
	int * frequency_spectrum;
	int ** frequency_age_spectrum;
	int * populations; //which populations are in SFS
	int * num_samples; //number of samples taken for each population
	int num_populations;
	int num_sites;
	int total_generations; //number of generations in the simulation

	sfs();
	~sfs();
};

#endif /* DATA_STRUCTS_H_ */
