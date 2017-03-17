/*
 * run.cu
 *
 *      Author: David Lawrie
 */

#include <fstream>
#include "go_fish.cuh"
#include "spectrum.h"

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

////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////

int main(int argc, char **argv) { run_validation_test(); }
