/*
 * run.cu
 *
 *      Author: David Lawrie
 */

#include "go_fish.cuh"
#include "run.h"

void run_migration_equilibrium_simulation(GO_Fish::allele_trajectories & a, float migration_rate){

	a.sim_input_constants.num_sites = 20*pow(10.f,7); 		//number of sites
	a.sim_input_constants.num_populations = 2;				//number of populations

	a.sim_input_constants.init_mse = false; 				//start from blank simulation
	Sim_Model::F_mu_h_constant mutation(pow(10.f,-9)); 		//per-site mutation rate
	Sim_Model::F_mu_h_constant inbreeding(1.f); 			//constant inbreeding
	Sim_Model::demography_constant demography(1000); 		//number of individuals in both populations
	std::cout<<migration_rate<<std::endl;
	Sim_Model::migration_constant_equal migration(migration_rate,a.sim_input_constants.num_populations); //constant migration rate
	Sim_Model::selection_constant selection(0.f); 			//constant selection coefficient
	Sim_Model::F_mu_h_constant dominance(0.f); 				//constant allele dominance (ignored as population is fully inbred)
	Sim_Model::bool_off dont_preserve; 						//don't preserve mutations
	Sim_Model::bool_off dont_sample; 						//don't sample generations
	a.sim_input_constants.compact_interval = 100;			//interval between compacts

	a.sim_input_constants.num_generations = 5*pow(10.f,4); //burn-in simulation, 50,0000 generations
	GO_Fish::run_sim(a,mutation,demography,migration,selection,inbreeding,dominance,dont_preserve,dont_sample); //only sample final generation

	GO_Fish::allele_trajectories c(a); 						//copy constructor, copies a to c (not actually needed for this simulation, just showing it is possible)

	Sim_Model::bool_on sample; 								//sample generation
	Sim_Model::bool_pulse<Sim_Model::bool_off,Sim_Model::bool_on> sample_strategy(dont_sample,sample,0,a.sim_input_constants.num_generations); //sample starting generation of second simulation (i.e. last generation of burn-in simulation)
	a.sim_input_constants.num_generations = pow(10.f,3);	//scenario simulation 1,0000 generations
	a.sim_input_constants.prev_sim_sample = 0; 				//start from previous simulation time sample 0
	GO_Fish::run_sim(a,mutation,demography,migration,selection,inbreeding,dominance,dont_preserve,sample_strategy,c); //scenario simulation, start from migration equilibrium, sample both start and final generations
}
