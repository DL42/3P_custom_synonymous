/*
 * run.cu
 *
 *      Author: David Lawrie
 */

#include "go_fish.cuh"
#include "spectrum.h"
#include "run.h"

void run_speed_test()
{
	//----- warm up scenario parameters -----
	GO_Fish::allele_trajectories a;
	a.sim_input_constants.num_generations = pow(10.f,5);//36;//50;//
	a.sim_input_constants.num_sites = 2*pow(10.f,7); //number of sites
	a.sim_input_constants.num_populations = 1; //number of populations
	a.sim_input_constants.seed1 = 0xbeeff00d; //random number seeds
	a.sim_input_constants.seed2 = 0xdecafbad;
	bool DFE = false;
	GO_Fish::const_parameter mutation(pow(10.f,-9)); //per-site mutation rate
	GO_Fish::const_parameter inbreeding(1.f); //constant inbreeding
	GO_Fish::const_demography demography(pow(10.f,5)*(1+inbreeding(0,0))); //number of individuals in population, set to maintain consistent effective number of chromosomes
	GO_Fish::const_equal_migration migration(0.f,a.sim_input_constants.num_populations); //constant migration rate
	float gamma = 0; //effective selection
	GO_Fish::const_selection selection(gamma/(2*demography(0,0))); //constant selection coefficient
	GO_Fish::const_parameter dominance(0.f); //constant allele dominance
	GO_Fish::do_nothing preserve; //don't preserve alleles from any generation
	GO_Fish::do_nothing sample_strategy; //only sample final generation
	//----- end warm up scenario parameters -----

	//----- warm up GPU -----
	GO_Fish::run_sim(a,mutation,demography,migration,selection,inbreeding,dominance,DFE,preserve,sample_strategy);
	std::cout<<std::endl<<"final number of mutations: " << a.maximal_num_mutations() << std::endl;

	//----- print allele counts x to x+y of warm up GPU scenario -----
	bool printSFS = true; //calculate and print out the SFS
	int start_index = 0;
	int print_num = 50;
	if(printSFS){
		SPECTRUM::sfs mySFS = SPECTRUM::site_frequency_spectrum(a,0,0,50);
		std::cout<< "allele count\t# mutations"<< std::endl;
		for(int printIndex = start_index; printIndex < min((mySFS.sample_size[0]-start_index),start_index+print_num); printIndex++){ std::cout<< (printIndex) << "\t" << mySFS.frequency_spectrum[printIndex] << std::endl;}
	}
	//----- end print allele counts x to x+y of warm up GPU scenario -----

	//GO_Fish::run_sim(a,mutation,demography,migration,selection,inbreeding,dominance,DFE,preserve,sample_strategy);
	GO_Fish::run_sim(a,mutation,demography,migration,selection,inbreeding,dominance,DFE,preserve,sample_strategy);
	//----- end warm up GPU -----

	//----- speed test scenario parameters -----
    cudaEvent_t start, stop;
    float elapsedTime;
    int num_iter = 10;
    a.sim_input_constants.compact_rate = 20;
    a.sim_input_constants.num_generations = pow(10.f,3);
    a.sim_input_constants.num_sites = 10*2*pow(10.f,7);
    a.sim_input_constants.seed1 = 0xbeeff00d; //random number seeds
    a.sim_input_constants.seed2 = 0xdecafbad;
	DFE = true;
	//----- end speed test scenario parameters -----

    //----- speed test -----
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for(int i = 0; i < num_iter; i++){
		GO_Fish::run_sim(a,mutation,demography,migration,selection,inbreeding,dominance,DFE,preserve,sample_strategy);
		if(i==0){ std::cout<< std::endl<<"final number of mutations: " << a.maximal_num_mutations() << std::endl; }
	}

	elapsedTime = 0;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("time elapsed: %f\n\n", elapsedTime/num_iter);
	//----- end speed test -----
	//
	cudaDeviceSynchronize();
	cudaDeviceReset();
}

////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////

void run_prev_sim_n_allele_traj_test(){
	GO_Fish::allele_trajectories a;
	a.sim_input_constants.num_generations = 5*pow(10.f,4);//36;//50;//
	a.sim_input_constants.num_sites = 2*pow(10.f,7); //number of sites
	a.sim_input_constants.num_populations = 1; //number of populations
	a.sim_input_constants.seed1 = 0xbeeff00d; //random number seeds
	a.sim_input_constants.seed2 = 0xdecafbad;
	a.sim_input_constants.init_mse = false;
	bool DFE = false;
	GO_Fish::const_parameter mutation1(1.07*pow(10.f,-9)); //per-site mutation rate
	GO_Fish::const_parameter inbreeding(1.f); //constant inbreeding
	GO_Fish::const_demography demography(pow(10.f,4)*(1+inbreeding(0,0))); //number of individuals in population, set to maintain consistent effective number of chromosomes
	GO_Fish::const_equal_migration migration(0.f,a.sim_input_constants.num_populations); //constant migration rate
	float gamma = 0; //effective selection
	GO_Fish::const_selection selection(gamma/(2*demography(0,0))); //constant selection coefficient
	GO_Fish::const_parameter dominance(0.f); //constant allele dominance
	GO_Fish::do_nothing dont_preserve; //don't preserve alleles
	GO_Fish::do_nothing dont_sample; //don't sample alleles
	GO_Fish::do_something sample; //sample alleles
	GO_Fish::pulse<GO_Fish::do_nothing,GO_Fish::do_something> sample_strategy(dont_sample,sample,0,a.sim_input_constants.num_generations); //sample starting generation of second simulation


	GO_Fish::run_sim(a,mutation1,demography,migration,selection,inbreeding,dominance,DFE,dont_preserve,dont_sample); //only sample final generation
	std::cout<<std::endl<<"final number of mutations: " << a.maximal_num_mutations() << std::endl;

	a.sim_input_constants.num_generations = pow(10.f,3);//36;//50;//
	a.sim_input_constants.prev_sim_sample = 0;
	GO_Fish::const_parameter mutation2(pow(10.f,-9)); //per-site mutation rate
	GO_Fish::run_sim(a,mutation2,demography,migration,selection,inbreeding,dominance,DFE,dont_preserve,sample_strategy,a);
	std::cout<<std::endl<<"number of time samples: " << a.num_time_samples();
	std::cout<<std::endl<<"starting number of mutations: " << a.num_mutations_time_sample(0) <<std::endl<<"final number of mutations: " << a.maximal_num_mutations() << std::endl;
	int mutation_range_begin = 0; int mutation_range_end = 10;
	std::cout<<"mutation IDs\tstart gen "<<a.sampled_generation(0)<<"\tfrequency\tfinal gen "<<a.final_generation()<<"\tfrequency"<<std::endl;
	for(int i = mutation_range_begin; i < mutation_range_end; i++){ std::cout<<"\t\t"<<a.mutation_ID_time_sample(0,i).toString()<<"\t"<<a.frequency(0,0,i)<<"\t\t"<<a.mutation_ID_all_samples(i).toString()<<"\t"<<a.frequency(1,0,i)<<std::endl; }
	mutation_range_begin = 11000; mutation_range_end = 11010;
	std::cout<<"mutation IDs\tID\tstart_frequency\tfinal_frequency"<<std::endl;
	for(int i = mutation_range_begin; i < mutation_range_end; i++){ std::cout<<"\t\t"<<a.mutation_ID_all_samples(i).toString()<<"\t"<<a.frequency(0,0,i)<<"\t"<<a.frequency(1,0,i)<<std::endl; }

	a.sim_input_constants.init_mse = true;
	a.sim_input_constants.seed1 = 0xdecafbad; //random number seeds
	a.sim_input_constants.seed2 = 0xbeeff00d;
	a.sim_input_constants.num_generations = pow(10.f,3);//36;//50;//
	GO_Fish::run_sim(a,mutation2,demography,migration,selection,inbreeding,dominance,DFE,dont_preserve,dont_sample);
	std::cout<<std::endl<<"final number of mutations: " << a.maximal_num_mutations() << std::endl;
}

////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////

double gx(double x, double gamma, double mu_site, double L){
	if(gamma != 0) return 2*mu_site*L*(1-exp(-1*gamma*(1-x)))/((1-exp(-1*gamma))*x*(1-x));
	return 2*mu_site*L/x;
}


double* G(double gamma,double mu_site, double L, double N_chrome){
	double total_SNPs = 0;
	double* g = new double[(int)N_chrome];

	for(int j = 1; j <= (N_chrome - 1); j++){
		double freq = j/(N_chrome);
		g[j] = gx(freq, gamma, mu_site, L);
		total_SNPs += g[j];
	}

	g[0] = L-total_SNPs;

	return g;
}

void run_validation_test(){

	GO_Fish::allele_trajectories b;
    float gamma = 0; //effective selection
	float h = 0.5; //dominance
	float F = 0.0; //inbreeding
	int N_ind = 0.03*pow(10.f,5)*(1+F);//300;// //bug at N_ind = 300, F =0.0, gamma = 0//number of individuals in population, set to maintain consistent effective number of chromosomes
	float s = gamma/(2*N_ind); //selection coefficient
	float mu = pow(10.f,-9); //per-site mutation rate
	int total_number_of_generations = pow(10.f,3);//0;//1000;//1;//36;//
	b.sim_input_constants.num_sites = 10*2*pow(10.f,7); //number of sites
	float m = 0.00; //migration rate
	b.sim_input_constants.num_populations = 1; //number of populations
	int num_iter = 50;
    bool DFE = false;
    b.sim_input_constants.compact_rate = 35;
    double* expectation = G(gamma,mu, b.sim_input_constants.num_sites, 2.0*N_ind/(1.0+F));
    double expected_total_SNPs = b.sim_input_constants.num_sites-expectation[0];

	for(int i = 0; i < num_iter; i++){
		b.sim_input_constants.seed1 = 0xbeeff00d + 2*i; //random number seeds
		b.sim_input_constants.seed2 = 0xdecafbad - 2*i;
		GO_Fish::run_sim(b, GO_Fish::const_parameter(mu), GO_Fish::const_demography(N_ind), GO_Fish::const_equal_migration(m,b.sim_input_constants.num_populations), GO_Fish::const_selection(s), GO_Fish::const_parameter(F), GO_Fish::const_parameter(h), DFE, GO_Fish::do_nothing(), GO_Fish::do_nothing());
		if(i==0){ std::cout<< "chi-gram number of mutations:"<<std::endl; }
		std::cout<< (int)expected_total_SNPs << "\t" << b.maximal_num_mutations() << "\t" << ((b.maximal_num_mutations() - expected_total_SNPs)/expected_total_SNPs) << std::endl;
	}

	delete [] expectation;
}
