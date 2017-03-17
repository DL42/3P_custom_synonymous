/*
 * run.cu
 *
 *      Author: David Lawrie
 */

#include <fstream>
#include "go_fish.cuh"
#include "spectrum.h"

void run_speed_test()
{
	//----- speed test scenario parameters -----
	GO_Fish::allele_trajectories a;
	a.sim_input_constants.num_populations = 1; //number of populations
	Sim_Model::F_mu_h_constant mutation(pow(10.f,-9)); //per-site mutation rate
	Sim_Model::F_mu_h_constant inbreeding(1.f); //constant inbreeding
	Sim_Model::demography_constant demography(pow(10.f,5)*(1+inbreeding(0,0))); //number of individuals in population, set to maintain consistent effective number of chromosomes
	Sim_Model::migration_constant_equal migration(0.f,a.sim_input_constants.num_populations); //constant migration rate
	float gamma = 0; //effective selection
	Sim_Model::selection_constant selection(gamma/(2*demography(0,0))); //constant selection coefficient (invariant wrt inbreeding and population size)
	Sim_Model::F_mu_h_constant dominance(0.f); //constant allele dominance
    a.sim_input_constants.num_generations = pow(10.f,3);
    a.sim_input_constants.seed1 = 0xbeeff00d; //random number seeds
    a.sim_input_constants.seed2 = 0xdecafbad;

    a.sim_input_constants.num_sites = 20*2*pow(10.f,7);
    a.sim_input_constants.compact_interval = 15;
	//----- end speed test scenario parameters -----

    //----- speed test -----
    cudaEvent_t start, stop;
    float elapsedTime;
    int num_iter = 20;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for(int i = 0; i < num_iter; i++){
		if(i == num_iter/2){ cudaEventRecord(start, 0); }
		GO_Fish::run_sim(a,mutation,demography,migration,selection,inbreeding,dominance,Sim_Model::bool_off(),Sim_Model::bool_off()); }

	elapsedTime = 0;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	std::cout<< std::endl<<"final number of mutations: " << a.maximal_num_mutations() << std::endl;
	printf("time elapsed: %f\n\n", 2*elapsedTime/num_iter);
	//----- end speed test -----
	//
}

////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////

int main(int argc, char **argv){ run_speed_test(); }
