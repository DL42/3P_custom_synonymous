/*
 * sim_results.h
 *
 *  Created on: Mar 20, 2016
 *      Author: dlawrie
 */

#ifndef SIM_RESULTS_H_
#define SIM_RESULTS_H_

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

#endif /* SIM_RESULTS_H_ */
