/*
 * run.cu
 *
 *      Author: David Lawrie
 */

#include <fstream>
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
	Sim_Model::F_mu_h_constant mutation(pow(10.f,-9)); //per-site mutation rate
	Sim_Model::F_mu_h_constant inbreeding(1.f); //constant inbreeding
	Sim_Model::demography_constant demography(pow(10.f,5)*(1+inbreeding(0,0))); //number of individuals in population, set to maintain consistent effective number of chromosomes
	Sim_Model::migration_constant_equal migration(0.f,a.sim_input_constants.num_populations); //constant migration rate
	float gamma = 0; //effective selection
	Sim_Model::selection_constant selection(gamma/(2*demography(0,0))); //constant selection coefficient
	Sim_Model::F_mu_h_constant dominance(0.f); //constant allele dominance
	Sim_Model::bool_off preserve; //don't preserve alleles from any generation
	Sim_Model::bool_off sample_strategy; //only sample final generation
	//----- end warm up scenario parameters -----

	//----- warm up GPU -----
	GO_Fish::run_sim(a,mutation,demography,migration,selection,inbreeding,dominance,preserve,sample_strategy);
	std::cout<<std::endl<<"final number of mutations: " << a.maximal_num_mutations() << std::endl;

	//----- print allele counts x to x+y of warm up GPU scenario -----
	bool printSFS = true; //calculate and print out the SFS
	int start_index = 0;
	int print_num = 50;
	Spectrum::SFS mySFS;
	if(printSFS){
		Spectrum::population_frequency_histogram(mySFS,a,0,0);
		std::cout<< "allele count\t# mutations"<< std::endl;
		for(int printIndex = start_index; printIndex < min((mySFS.sample_size[0]-start_index),start_index+print_num); printIndex++){ std::cout<< (printIndex) << "\t" << mySFS.frequency_spectrum[printIndex] << std::endl;}
	}
	//----- end print allele counts x to x+y of warm up GPU scenario -----

	//GO_Fish::run_sim(a,mutation,demography,migration,selection,inbreeding,dominance,DFE,preserve,sample_strategy);
	GO_Fish::run_sim(a,mutation,demography,migration,selection,inbreeding,dominance,preserve,sample_strategy);
	//----- end warm up GPU -----

	//----- speed test scenario parameters -----
    cudaEvent_t start, stop;
    float elapsedTime;
    int num_iter = 10;
    a.sim_input_constants.compact_interval = 35;
    a.sim_input_constants.num_generations = pow(10.f,3);
    a.sim_input_constants.num_sites = 2*pow(10.f,7);
    a.sim_input_constants.seed1 = 0xbeeff00d; //random number seeds
    a.sim_input_constants.seed2 = 0xdecafbad;
	//----- end speed test scenario parameters -----

    //----- speed test -----
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for(int i = 0; i < num_iter; i++){ GO_Fish::run_sim(a,mutation,demography,migration,selection,inbreeding,dominance,Sim_Model::bool_off(),Sim_Model::bool_off()); }

	elapsedTime = 0;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	std::cout<< std::endl<<"final number of mutations: " << a.maximal_num_mutations() << std::endl;
	std::cout<< std::endl<<a.num_time_samples()<<std::endl;
	for(int i = 0; i < a.num_time_samples(); i++){
		std::cout<<a.mutation_ID(1).toString()<<" "<<a.frequency(i,0,1)<<"\t"<<a.mutation_ID(50000).toString()<<" "<<a.frequency(i,0,50000)<<"\t"<<a.mutation_ID(100000).toString()<<" "<<a.frequency(i,0,100000)<<std::endl;
	}

	std::cout<<std::endl;

	printf("time elapsed: %f\n\n", elapsedTime/num_iter);
	//----- end speed test -----
	//
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
	float h = 0.5; //dominance
	float F = 0.0; //inbreeding
	int N_ind = pow(10.f,5);//300;// //bug at N_ind = 300, F =0.0/1.0, gamma = 0//number of individuals in population, set to maintain consistent effective number of chromosomes across all inbreeding coefficients
    float gamma = 0*(1+F); //effective selection //set to maintain consistent level of selection across all inbreeding coefficients for the same effective number of chromosomes, drift and selection are invariant with respect to inbreeding
	float mu = pow(10.f,-9); //per-site mutation rate
	int total_number_of_generations = pow(10.f,3);//0;//1000;//1;//36;//
	b.sim_input_constants.num_generations = total_number_of_generations;
	b.sim_input_constants.num_sites = 20*2*pow(10.f,7); //number of sites
	float m = 0.00; //migration rate
	b.sim_input_constants.num_populations = 1; //number of populations
	int num_iter = 50;
    b.sim_input_constants.compact_interval = 20;
   // double* expectation = G(gamma,mu, b.sim_input_constants.num_sites, 2.0*N_ind/(1.0+F));
    //double expected_total_SNPs = b.sim_input_constants.num_sites-expectation[0];
    Spectrum::SFS * my_spectra = new Spectrum::SFS[num_iter];

    cudaEvent_t start, stop;
    float elapsedTime;
    int sample_size = 200;
	for(int i = 0; i < num_iter; i++){
		if(i == round(num_iter/2.f)){
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0);
		}

		b.sim_input_constants.seed1 = 0xbeeff00d + 2*i; //random number seeds
		b.sim_input_constants.seed2 = 0xdecafbad - 2*i;
		GO_Fish::run_sim((b), Sim_Model::F_mu_h_constant(mu), Sim_Model::demography_constant(N_ind), Sim_Model::migration_constant_equal(m,b.sim_input_constants.num_populations), Sim_Model::selection_constant(gamma,Sim_Model::demography_constant(N_ind),Sim_Model::F_mu_h_constant(F)), Sim_Model::F_mu_h_constant(F), Sim_Model::F_mu_h_constant(h), Sim_Model::bool_off(), Sim_Model::bool_off());
		Spectrum::site_frequency_spectrum(my_spectra[i],(b),0,0,sample_size);
		//if(i==0){ std::cout<< "dispersion/chi-gram of number of mutations:"<<std::endl; }
		//std::cout<<b.maximal_num_mutations()<<std::endl;
		//std::cout<< (int)expected_total_SNPs << "\t" << b.maximal_num_mutations() << "\t" << ((b.maximal_num_mutations() - expected_total_SNPs)/expected_total_SNPs) << std::endl;
	}

	elapsedTime = 0;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	std::cout<<"\ntime elapsed: "<< 2*elapsedTime/num_iter<<std::endl;
	//----- end speed test -----
	//
	//if(my_spectra[0].frequency_spectrum[0] < 0){ std::cout<<std::endl<<0<<"\t"<<my_spectra[0].frequency_spectrum[0]<<std::endl; }
	std::cout<<std::endl<<"SFS :"<<std::endl<< "allele count\tavg# mutations\tstandard dev\tcoeff of variation (aka relative standard deviation)"<< std::endl;
	float avg_num_mutations = 0;
	for(int i = 1; i < sample_size; i++){
		double avg = 0;
		double std = 0;
		float num_mutations;
		for(int j = 0; j < num_iter; j++){ num_mutations = my_spectra[j].num_mutations; avg += my_spectra[j].frequency_spectrum[i]/(num_iter*num_mutations); if(i==1){ avg_num_mutations += ((float)num_mutations)/num_iter; }}
		for(int j = 0; j < num_iter; j++){ num_mutations = my_spectra[j].num_mutations; std += 1.0/(num_iter-1)*pow(my_spectra[j].frequency_spectrum[i]/num_mutations-avg,2); }
		std = sqrt(std);
		std::cout<<i<<"\t"<<avg<<"\t"<<std<<"\t"<<(std/avg)<<std::endl;
		//std::cout<<avg<<std::endl;
	}
	std::cout<<avg_num_mutations<<std::endl;
	//delete [] expectation;
	delete [] my_spectra;
}
