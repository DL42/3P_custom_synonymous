/*
 * run.cu
 *
 *      Author: David Lawrie
 */

#include "go_fish.cuh"
#include "spectrum.h"
#include <vector>

void run_validation_test(){
	typedef Sim_Model::demography_constant dem_const;
	typedef Sim_Model::demography_population_specific<dem_const,dem_const> dem_pop_const;
	typedef Sim_Model::demography_piecewise<dem_pop_const,dem_pop_const> init_expansion;
	typedef Sim_Model::demography_exponential_growth exp_growth;
	typedef Sim_Model::demography_population_specific<dem_const,exp_growth> dem_pop_const_exp;

	typedef Sim_Model::migration_constant_equal mig_const;
	typedef Sim_Model::migration_constant_directional<mig_const> mig_dir;
	typedef Sim_Model::migration_constant_directional<mig_dir> mig_split;
	typedef Sim_Model::migration_piecewise<mig_const,mig_split> split_pop0;

	GO_Fish::allele_trajectories b;
	b.sim_input_constants.num_populations = 2; 				//number of populations
	b.sim_input_constants.num_generations = pow(10.f,3)+1;	//1,000 generations

	Sim_Model::F_mu_h_constant codominant(0.5f); 			//dominance (co-dominant)
	Sim_Model::F_mu_h_constant outbred(0.f); 				//inbreeding (outbred)
	Sim_Model::F_mu_h_constant mutation(pow(10.f,-9)); 		//per-site mutation rate 10^-9

	int N_ind = pow(10.f,4)*(1+outbred(0,0));				//initial number of individuals in population, set to maintain consistent effective number of chromosomes across all inbreeding coefficients
	dem_const pop0(N_ind);
	dem_const pop1(0);
	dem_pop_const gen0(pop0,pop1,1);
	dem_const pop0_final(2*N_ind);
	dem_pop_const gen1(pop0_final,pop1,1);
	init_expansion gen_0_1(gen0,gen1,1);
	exp_growth pop1_gen100((log(100.f)/(900.f)),0.05*N_ind,100);
	//exp_growth pop1_gen100((log(1.f)/900.f),5*N_ind,100);
	dem_pop_const_exp gen100(pop0_final,pop1_gen100,1);
	Sim_Model::demography_piecewise<init_expansion,dem_pop_const_exp> demography_model(gen_0_1,gen100,100);
	//for(int i = 0; i < 1000; i++){ std::cout<< demography_model(0,i) << "\t" << demography_model(1,i) << std::endl; }

	mig_const no_mig_pop0;
	mig_dir no_pop1_gen0(0.f,1,1,no_mig_pop0);
	mig_split create_pop1(1.f,0,1,no_pop1_gen0);
	split_pop0 migration_split(no_mig_pop0,create_pop1,100);
	float mig = 1.f/(2.f*N_ind);
	mig_const mig_prop(mig,b.sim_input_constants.num_populations);	//migration rate
	Sim_Model::migration_piecewise<split_pop0,mig_const> mig_model(migration_split,mig_prop,100+1);
	//for(int i = 0; i < 1000; i++){ std::cout<< mig_model(0,0,i) << "\t" << mig_model(0,1,i) << "\t" << mig_model(1,0,i) << "\t" << mig_model(1,1,i) << std::endl; }

	float gamma = -4*(1+outbred(0,0)); 								//effective selection //set to maintain consistent level of selection across all inbreeding coefficients for the same effective number of chromosomes, drift and selection are invariant with respect to inbreeding
	Sim_Model::selection_constant weak_del(gamma,demography_model,outbred);

	b.sim_input_constants.compact_interval = 30;			//compact interval
	b.sim_input_constants.num_sites = 100*2*pow(10.f,7); 	//number of sites
	int sample_size = 1000;									//number of samples in SFS

	int num_iter = 50;										//number of iterations
    Spectrum::SFS my_spectra;

    cudaEvent_t start, stop;								//CUDA timing functions
    float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float avg_num_mutations = 0;
	float avg_num_mutations_sim = 0;
	std::vector<std::vector<float> > results(num_iter); 	//storage for SFS results
	for(int j = 0; j < num_iter; j++){ results[j].reserve(sample_size); }

	for(int j = 0; j < num_iter; j++){
		if(j == num_iter/2){ cudaEventRecord(start, 0); } 	//use 2nd half of the simulations to time simulation runs + SFS creation

		b.sim_input_constants.seed1 = 0xbeeff00d + 2*j; 	//random number seeds
		b.sim_input_constants.seed2 = 0xdecafbad - 2*j;
		GO_Fish::run_sim(b, mutation, demography_model, mig_model, weak_del, outbred, codominant, Sim_Model::bool_off(), Sim_Model::bool_off());
		Spectrum::site_frequency_spectrum(my_spectra,b,0,1,sample_size);

		avg_num_mutations += ((float)my_spectra.num_mutations)/num_iter;
		avg_num_mutations_sim += b.maximal_num_mutations()/num_iter;
		for(int i = 0; i < sample_size; i++){ results[j][i] = my_spectra.frequency_spectrum[i]; /*if(i == 1){ std::cout<< results[j][i]/(b.num_sites() - results[j][0]) << std::endl; } */}
		//std::cout<<std::endl;
	}

	elapsedTime = 0;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	std::cout<<std::endl<<"SFS :"<<std::endl<< "allele count\tavg# mutations\tstandard dev\tcoeff of variation (aka relative standard deviation)"<< std::endl;
	for(int i = 1; i < sample_size; i++){
		double avg = 0;
		double std = 0;
		float num_mutations;
		for(int j = 0; j < num_iter; j++){ num_mutations = b.num_sites() - results[j][0]; avg += results[j][i]/(num_iter*num_mutations); }
		for(int j = 0; j < num_iter; j++){ num_mutations = b.num_sites() - results[j][0]; std += 1.0/(num_iter-1)*pow(results[j][i]/num_mutations-avg,2); }
		std = sqrt(std);
		std::cout<<i<<"\t"<<avg<<"\t"<<std<<"\t"<<(std/avg)<<std::endl;
	}
	std::cout<<avg_num_mutations<<"\t"<<avg_num_mutations_sim<<std::endl;
	std::cout<<"\ntime elapsed (ms): "<< 2*elapsedTime/num_iter<<std::endl;
}

void run_validation_test_simple(){
	typedef Sim_Model::demography_constant dem_const;
	typedef Sim_Model::demography_piecewise<dem_const,dem_const> init_expansion;

	GO_Fish::allele_trajectories b;
	b.sim_input_constants.num_populations = 1; 				//number of populations
	b.sim_input_constants.num_generations = pow(10.f,3)+1;	//1,000 generations

	Sim_Model::F_mu_h_constant codominant(0.5f); 			//dominance (co-dominant)
	Sim_Model::F_mu_h_constant outbred(0.f); 				//inbreeding (outbred)
	Sim_Model::F_mu_h_constant mutation(pow(10.f,-9)); 		//per-site mutation rate 10^-9
	Sim_Model::migration_constant_equal no_mig;				//no migration

	int N_ind = pow(10.f,4)*(1+outbred(0,0));				//initial number of individuals in population, set to maintain consistent effective number of chromosomes across all inbreeding coefficients
	dem_const pop0(N_ind);
	dem_const pop0_final(5*N_ind);
	init_expansion demography_model(pop0,pop0_final,1);



	float gamma = -4*(1+outbred(0,0)); 								//effective selection //set to maintain consistent level of selection across all inbreeding coefficients for the same effective number of chromosomes, drift and selection are invariant with respect to inbreeding
	Sim_Model::selection_constant weak_del(gamma,demography_model,outbred);

	b.sim_input_constants.compact_interval = 30;			//compact interval
	b.sim_input_constants.num_sites = 100*2*pow(10.f,7); 	//number of sites
	int sample_size = 1000;									//number of samples in SFS

	int num_iter = 50;										//number of iterations
    Spectrum::SFS my_spectra;

    cudaEvent_t start, stop;								//CUDA timing functions
    float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float avg_num_mutations = 0;
	float avg_num_mutations_sim = 0;
	std::vector<std::vector<float> > results(num_iter); 	//storage for SFS results
	for(int j = 0; j < num_iter; j++){ results[j].reserve(sample_size); }

	for(int j = 0; j < num_iter; j++){
		if(j == num_iter/2){ cudaEventRecord(start, 0); } 	//use 2nd half of the simulations to time simulation runs + SFS creation

		b.sim_input_constants.seed1 = 0xbeeff00d + 2*j; 	//random number seeds
		b.sim_input_constants.seed2 = 0xdecafbad - 2*j;
		GO_Fish::run_sim(b, mutation, demography_model, no_mig, weak_del, outbred, codominant, Sim_Model::bool_off(), Sim_Model::bool_off());
		Spectrum::site_frequency_spectrum(my_spectra,b,0,0,sample_size);

		avg_num_mutations += ((float)my_spectra.num_mutations)/num_iter;
		avg_num_mutations_sim += b.maximal_num_mutations()/num_iter;
		for(int i = 0; i < sample_size; i++){ results[j][i] = my_spectra.frequency_spectrum[i]; /*if(i == 1){ std::cout<< results[j][i]/(b.num_sites() - results[j][0]) << std::endl; } */}
		//std::cout<<std::endl;
	}

	elapsedTime = 0;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	std::cout<<std::endl<<"SFS :"<<std::endl<< "allele count\tavg# mutations\tstandard dev\tcoeff of variation (aka relative standard deviation)"<< std::endl;
	for(int i = 1; i < sample_size; i++){
		double avg = 0;
		double std = 0;
		float num_mutations;
		for(int j = 0; j < num_iter; j++){ num_mutations = b.num_sites() - results[j][0]; avg += results[j][i]/(num_iter*num_mutations); }
		for(int j = 0; j < num_iter; j++){ num_mutations = b.num_sites() - results[j][0]; std += 1.0/(num_iter-1)*pow(results[j][i]/num_mutations-avg,2); }
		std = sqrt(std);
		std::cout<<i<<"\t"<<avg<<"\t"<<std<<"\t"<<(std/avg)<<std::endl;
	}
	std::cout<<avg_num_mutations<<"\t"<<avg_num_mutations_sim<<std::endl;
	std::cout<<"\ntime elapsed (ms): "<< 2*elapsedTime/num_iter<<std::endl;
}

////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
//	run_validation_test();
	run_validation_test_simple();
}
