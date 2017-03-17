/*
 * run.cu
 *
 *      Author: David Lawrie
 */

#include <fstream>
#include "go_fish.cuh"
#include "run.h"

void run_prev_sim_n_allele_traj_test(){
	GO_Fish::allele_trajectories a;
	a.sim_input_constants.num_generations = 5*pow(10.f,4);//36;//50;//
	a.sim_input_constants.num_sites = 2*pow(10.f,7); //number of sites
	a.sim_input_constants.num_populations = 1; //number of populations
	a.sim_input_constants.seed1 = 0xbeeff00d; //random number seeds
	a.sim_input_constants.seed2 = 0xdecafbad;
	a.sim_input_constants.init_mse = false;
	Sim_Model::F_mu_h_constant mutation1(1.07*pow(10.f,-9)); //per-site mutation rate
	Sim_Model::F_mu_h_constant inbreeding(1.f); //constant inbreeding
	Sim_Model::demography_constant demography(pow(10.f,4)*(1+inbreeding(0,0))); //number of individuals in population, set to maintain consistent effective number of chromosomes
	Sim_Model::migration_constant_equal migration(0.f,a.sim_input_constants.num_populations); //constant migration rate
	float gamma = 0; //effective selection
	Sim_Model::selection_constant selection(gamma/(2*demography(0,0))); //constant selection coefficient
	Sim_Model::F_mu_h_constant dominance(0.f); //constant allele dominance
	Sim_Model::bool_off dont_preserve; //don't preserve alleles
	Sim_Model::bool_off dont_sample; //don't sample alleles
	Sim_Model::bool_on sample; //sample alleles
	Sim_Model::bool_pulse<Sim_Model::bool_off,Sim_Model::bool_on> sample_strategy(dont_sample,sample,0,a.sim_input_constants.num_generations); //sample starting generation of second simulation


	GO_Fish::run_sim(a,mutation1,demography,migration,selection,inbreeding,dominance,dont_preserve,dont_sample); //only sample final generation
	std::cout<<std::endl<<"final number of mutations: " << a.maximal_num_mutations() << std::endl;

	GO_Fish::allele_trajectories c(a);

	a.sim_input_constants.num_generations = pow(10.f,3);//36;//50;//
	a.sim_input_constants.prev_sim_sample = 0;
	Sim_Model::F_mu_h_constant mutation2(pow(10.f,-9)); //per-site mutation rate
	GO_Fish::run_sim(a,mutation2,demography,migration,selection,inbreeding,dominance,dont_preserve,sample_strategy,c);

	std::cout<<std::endl<<"number of time samples: " << a.num_time_samples();
	std::cout<<std::endl<<"starting number of mutations: " << a.num_mutations_time_sample(0) <<std::endl<<"final number of mutations: " << a.maximal_num_mutations() << std::endl;
	int mutation_range_begin = 0; int mutation_range_end = 10;
	std::cout<<"mutation IDs\tstart gen "<<a.sampled_generation(0)<<"\tfrequency\tfinal gen "<<a.final_generation()<<"\tfrequency"<<std::endl;
	for(int i = mutation_range_begin; i < mutation_range_end; i++){ std::cout<<"\t\t"<<a.mutation_ID(i)<<"\t"<<a.frequency(0,0,i)<<"\t\t"<<a.mutation_ID(i)<<"\t"<<a.frequency(1,0,i)<<std::endl; }
	mutation_range_begin = 11000; mutation_range_end = 11010;
	std::cout<<"mutation IDs\tID\tstart_frequency\tfinal_frequency"<<std::endl;
	for(int i = mutation_range_begin; i < mutation_range_end; i++){ std::cout<<"\t\t"<<a.mutation_ID(i)<<"\t"<<a.frequency(0,0,i)<<"\t"<<a.frequency(1,0,i)<<std::endl; }

	GO_Fish::allele_trajectories b = a; //tests both copy-constructor and copy-assignment

	std::cout<<std::endl<<"number of time samples: " << b.num_time_samples();
	std::cout<<std::endl<<"starting number of mutations: " << b.num_mutations_time_sample(0) <<std::endl<<"final number of mutations: " << b.maximal_num_mutations() << std::endl;
	mutation_range_begin = 0; mutation_range_end = 10;
	std::cout<<"mutation IDs\tstart gen "<<b.sampled_generation(0)<<"\tfrequency\tfinal gen "<<b.final_generation()<<"\tfrequency"<<std::endl;
	for(int i = mutation_range_begin; i < mutation_range_end; i++){ std::cout<<"\t\t"<<b.mutation_ID(i)<<"\t"<<b.frequency(0,0,i)<<"\t\t"<<b.mutation_ID(i)<<"\t"<<b.frequency(1,0,i)<<std::endl; }
	mutation_range_begin = 11000; mutation_range_end = 11010;
	std::cout<<"mutation IDs\tID\tstart_frequency\tfinal_frequency"<<std::endl;
	for(int i = mutation_range_begin; i < mutation_range_end; i++){ std::cout<<"\t\t"<<b.mutation_ID(i)<<"\t"<<b.frequency(0,0,i)<<"\t"<<b.frequency(1,0,i)<<std::endl; }

	std::ofstream outfile;
	outfile.open("afile.dat");
	outfile<<a;
	outfile.close();
	outfile.open("bfile.dat");
	outfile<<b;
	outfile.close();

	a.sim_input_constants.init_mse = true;
	a.sim_input_constants.seed1 = 0xdecafbad; //random number seeds
	a.sim_input_constants.seed2 = 0xbeeff00d;
	a.sim_input_constants.num_generations = pow(10.f,3);//36;//50;//
	GO_Fish::run_sim(a,mutation2,demography,migration,selection,inbreeding,dominance,dont_preserve,dont_sample);
	std::cout<<std::endl<<"final number of mutations: " << a.maximal_num_mutations() << std::endl;
}
