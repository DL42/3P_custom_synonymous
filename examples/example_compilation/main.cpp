/*
 * main.cpp
 *
 *      Author: David Lawrie
 */

#include "run.h"
#include <fstream>

int main(int argc, char **argv){
	GO_Fish::allele_trajectories a;
	a.sim_input_constants.seed1 = 0xbeeff00d; //random number seeds
	a.sim_input_constants.seed2 = 0xdecafbad;

	float migration_rate = 0.3;
	run_migration_equilibrium_simulation(a,migration_rate); 	//run simulation
	GO_Fish::allele_trajectories b = a; 						//copy-assignment (unneeded, just showing it is possible)
	a.reset(); 													//frees memory held by a (unneeded, just showing it is possible)

	/* --- output simulation information --- */
	std::cout<<std::endl<<"number of time samples: " << b.num_time_samples();
	std::cout<<std::endl<<"mutations in first time sample: " << b.num_mutations_time_sample(0) <<std::endl<<"mutations in final time sample: " << b.maximal_num_mutations() << std::endl; //mutations in final time sample >= mutations in first time sample as all mutations in the latter were preserved by sampling
	int mutation_range_begin = 0; int mutation_range_end = 10;
	std::cout<<"ID\tstart frequency pop 1\tstart frequency pop 2\tfinal frequency pop 1\tfinal frequency pop 2"<<std::endl;
	for(int i = mutation_range_begin; i < mutation_range_end; i++){ std::cout<<b.mutation_ID(i)<<"\t"<<b.frequency(0,0,i)<<"\t"<<b.frequency(0,1,i)<<"\t"<<b.frequency(1,0,i)<<"\t"<<b.frequency(1,1,i)<<std::endl; }
	mutation_range_begin = 26000; mutation_range_end = 26010;
	std::cout<<"ID\tstart frequency pop 1\tstart frequency pop 2\tfinal frequency pop 1\tfinal frequency pop 2"<<std::endl;
	for(int i = mutation_range_begin; i < mutation_range_end; i++){ std::cout<<b.mutation_ID(i)<<"\t"<<b.frequency(0,0,i)<<"\t"<<b.frequency(0,1,i)<<"\t"<<b.frequency(1,0,i)<<"\t"<<b.frequency(1,1,i)<<std::endl; }

	std::ofstream outfile;
	outfile.open("bfile.dat");
	outfile<<b;
	outfile.close();
	/* --- end output simulation information --- */
}
