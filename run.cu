/*
 * run.cu
 *
 *      Author: David Lawrie
 */

//currently using separate compilation which is a little slower than whole program compilation because a Rand1 function is not inlined and used in multiple sources (objects)
#include "go_fish.h"
#include "sfs.h"
#include "run.h"

using namespace std;
using namespace GO_Fish;
using namespace SFS;

void run_speed_test()
{
	//----- warm up scenario parameters -----
	allele_trajectories a;
    float gamma = 0; //effective selection
	float h = 0.5; //dominance
	float F = 0.0; //inbreeding
	int N_ind = pow(10.f,5)*(1+F); //number of individuals in population, set to maintain consistent effective number of chromosomes
	float s = gamma/(2*N_ind); //selection coefficient
	float mu = pow(10.f,-9); //per-site mutation rate
	a.num_generations = pow(10.f,5);//36;//50;//
	a.num_sites = 2*pow(10.f,7); //number of sites
	float m = 0.00; //migration rate
	a.num_populations = 1; //number of populations
	a.seed1 = 0xbeeff00d; //random number seeds
	a.seed2 = 0xdecafbad;
	bool DFE = false;

	//----- end warm up scenario parameters -----

	//----- warm up GPU -----
	bool printSFS = true; //calculate and print out the SFS
	run_GO_Fish_sim(&a,const_parameter(mu), const_demography(N_ind), const_equal_migration(m,a.num_populations), const_selection(s), const_parameter(F), const_parameter(h), DFE, DFE, do_nothing(), do_nothing());
	cout<<endl<<"final number of mutations: " << a.time_samples[0]->num_mutations << endl;

	//----- print allele counts x to x+y of warm up GPU scenario -----
	int start_index = 0;
	int print_num = 50;
	if(printSFS){
		sfs mySFS = site_frequency_spectrum(a.time_samples[0],0);
		cout<< "allele count\t# mutations"<< endl;
		for(int printIndex = start_index; printIndex < min((mySFS.num_samples[0]-start_index),start_index+print_num); printIndex++){ cout<< (printIndex) << "\t" << mySFS.frequency_spectrum[printIndex] <<endl;}
	}
	//----- end print allele counts x to x+y of warm up GPU scenario -----

	run_GO_Fish_sim(&a, const_parameter(mu), const_demography(N_ind), const_equal_migration(m,a.num_populations), const_selection(s), const_parameter(F), const_parameter(h), DFE, DFE, do_nothing(), do_nothing());
	//----- end warm up GPU -----

	//----- speed test scenario parameters -----
    cudaEvent_t start, stop;
    float elapsedTime;
    int num_iter = 10;
    a.compact_rate = 35;

    gamma = 0;
    h = 0.0;
    F = 1.0;
    N_ind = pow(10.f,5)*(1+F);
    s = gamma/(2*N_ind);
    mu = pow(10.f,-9);
    a.num_generations = pow(10.f,3);
    a.num_sites = 1*2*pow(10.f,7);
    a.num_populations = 1;
    m = 0.0;
    a.seed1 = 0xbeeff00d; //random number seeds
    a.seed2 = 0xdecafbad;
	DFE = true;
	//----- end speed test scenario parameters -----

    //----- speed test -----
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for(int i = 0; i < num_iter; i++){
		run_GO_Fish_sim(&a, const_parameter(mu), const_demography(N_ind), const_equal_migration(m,a.num_populations), const_selection(s), const_parameter(F), const_parameter(h), DFE, DFE, do_nothing(), do_nothing());
		if(i==0){ cout<<endl<<"final number of mutations: " << a.time_samples[0]->num_mutations << endl; }
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

	allele_trajectories b;
    float gamma = 0; //effective selection
	float h = 0.5; //dominance
	float F = 0.0; //inbreeding
	int N_ind = 0.03*pow(10.f,5)*(1+F);//300;// //bug at N_ind = 300, F =0.0, gamma = 0//number of individuals in population, set to maintain consistent effective number of chromosomes
	float s = gamma/(2*N_ind); //selection coefficient
	float mu = pow(10.f,-9); //per-site mutation rate
	int total_number_of_generations = pow(10.f,3);//0;//1000;//1;//36;//
	b.num_sites = 10*2*pow(10.f,7); //number of sites
	float m = 0.00; //migration rate
	b.num_populations = 1; //number of populations
	int num_iter = 50;
    bool DFE = false;
    b.compact_rate = 35;
    double* expectation = G(gamma,mu, b.num_sites, 2.0*N_ind/(1.0+F));
    double expected_total_SNPs = b.num_sites-expectation[0];

	for(int i = 0; i < num_iter; i++){
		b.seed1 = 0xbeeff00d + 2*i; //random number seeds
		b.seed2 = 0xdecafbad - 2*i;
		run_GO_Fish_sim(&b, const_parameter(mu), const_demography(N_ind), const_equal_migration(m,b.num_populations), const_selection(s), const_parameter(F), const_parameter(h), DFE, DFE, do_nothing(), do_nothing());
		if(i==0){ cout<< "chi-gram number of mutations:"<<endl; }
		cout<< (int)expected_total_SNPs << "\t" << b.time_samples[0]->num_mutations<< "\t" << ((b.time_samples[0]->num_mutations - expected_total_SNPs)/expected_total_SNPs) << endl;
	}

	delete [] expectation;
}
