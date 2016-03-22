/*
 * data_structs.h
 *
 *      Author: David Lawrie
 */

#ifndef DATA_STRUCTS_H_
#define DATA_STRUCTS_H_

#include <stdio.h>
#include <cuda_runtime.h>

/* ----- cuda error checking ----- */
#define __DEBUG__ false
#define cudaCheckErrors(expr1,expr2,expr3) { cudaError_t e = expr1; int g = expr2; int p = expr3; if (e != cudaSuccess) { fprintf(stderr,"error %d %s\tfile %s\tline %d\tgeneration %d\t population %d\n", e, cudaGetErrorString(e),__FILE__,__LINE__, g,p); exit(1); } }
#define cudaCheckErrorsAsync(expr1,expr2,expr3) { cudaCheckErrors(expr1,expr2,expr3); if(__DEBUG__){ cudaCheckErrors(cudaDeviceSynchronize(),expr2,expr3); } }
/* ----- end of cuda error checking ----- */

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
	~sim_result();
};

#endif /* DATA_STRUCTS_H_ */
