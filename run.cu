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
    float gamma = 0; //effective selection
	float h = 0.5; //dominance
	float F = 0.0; //inbreeding
	int N_ind = pow(10.f,5)*(1+F); //number of individuals in population, set to maintain consistent effective number of chromosomes
	float s = gamma/(2*N_ind); //selection coefficient
	float mu = pow(10.f,-9); //per-site mutation rate
	int total_number_of_generations = pow(10.f,5);//36;//50;//
	float L = 2*pow(10.f,7); //number of sites
	float m = 0.00; //migration rate
	int num_pop = 1; //number of populations
	int seed1 = 0xbeeff00d; //random number seeds
	int seed2 = 0xdecafbad;
	bool printSFS = true; //calculate and print out the SFS
	//----- end warm up scenario parameters -----

	//----- warm up GPU -----
	sim_result * a = run_sim(const_parameter(mu), const_demography(N_ind), const_equal_migration(m,num_pop), const_selection(s), const_parameter(F), const_parameter(h), total_number_of_generations, L, num_pop, seed1, seed2, do_nothing(), do_nothing(), 0, true);
	cout<<endl<<"final number of mutations: " << a[0].num_mutations << endl;

	//----- print allele counts x to x+y of warm up GPU scenario -----
	int start_index = 0;
	int print_num = 50;
	if(printSFS){
		sfs mySFS = site_frequency_spectrum(a[0],0);
		cout<< "allele count\t# mutations"<< endl;
		for(int printIndex = start_index; printIndex < min((mySFS.num_samples[0]-start_index),start_index+print_num); printIndex++){ cout<< (printIndex) << "\t" << mySFS.frequency_spectrum[printIndex] <<endl;}
	}
	//----- end print allele counts x to x+y of warm up GPU scenario -----
	delete [] a;

	a = run_sim(const_parameter(mu), const_demography(N_ind), const_equal_migration(m,num_pop), const_selection(s), const_parameter(F), const_parameter(h), total_number_of_generations, L, num_pop, seed1, seed2, do_nothing(), do_nothing(), 0, true);
	delete [] a;
	//----- end warm up GPU -----

	//----- speed test scenario parameters -----
    cudaEvent_t start, stop;
    float elapsedTime;
    int num_iter = 10;
    int compact_rate = 35;

    gamma = 0;
    h = 0.0;
    F = 1.0;
    N_ind = pow(10.f,5)*(1+F);
    s = gamma/(2*N_ind);
    mu = pow(10.f,-9);
    total_number_of_generations = pow(10.f,3);
    L = 1*2*pow(10.f,7);
    num_pop = 1;
    m = 0.0;
	seed1 = 0xbeeff00d; //random number seeds
	seed2 = 0xdecafbad;
	//----- end speed test scenario parameters -----

    //----- speed test -----
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for(int i = 0; i < num_iter; i++){
		sim_result * b = run_sim(const_parameter(mu), const_demography(N_ind), const_equal_migration(m,num_pop), const_selection(s), const_parameter(F), const_parameter(h), total_number_of_generations, L, num_pop, seed1, seed2, do_nothing(), do_nothing(), 0, true, sim_result(), compact_rate);
		if(i==0){ cout<<endl<<"final number of mutations: " << b[0].num_mutations << endl; }
		delete [] b;
	}

	elapsedTime = 0;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("time elapsed: %f\n\n", elapsedTime/num_iter);
	//----- end speed test -----

	cudaDeviceSynchronize();
	cudaDeviceReset();
}

/*    gx = @(x,gamma,mu_site,L)2*mu_site*L*(1-exp(-1*gamma*(1-x)))/((1-exp(-1*gamma))*x*(1-x));
    function [total_SNPs,g] = m(gamma,mu_site,L,Npop)
        total_SNPs = 0;
        g = zeros((2*Npop-1),1);
        for j = 1:(2*Npop-1)
            freq = j/(2*Npop);
            if(gamma ~= 0)
                g(j) = gx(freq,gamma,mu_site,L);
            else
                g(j) = 2*mu_site*L/freq;
            end
            total_SNPs = total_SNPs + g(j);
        end
    end*/

float gx(float x, float gamma, float mu_site, float L){
	if(gamma != 0) return 2*mu_site*L*(1-exp(-1*gamma*(1-x)))/((1-exp(-1*gamma))*x*(1-x));
	return 2*mu_site*L/x;
}

struct G_result{
	float * g;
	float total_SNPs;
};

G_result G(float gamma,float mu_site, float L, float N_chrome){
	float total_SNPs = 0;
	float * g = new float[(int)N_chrome-1];

	for(int j = 1; j <= (N_chrome - 1); j++){
		float freq = j/(N_chrome);
		g[j-1] = gx(freq, gamma, mu_site, L);
		total_SNPs += g[j-1];
	}

	G_result r;
	r.g = g;
	r.total_SNPs = total_SNPs;

	return r;
}

void run_validation_test(){
    float gamma = -10; //effective selection
	float h = 0.5; //dominance
	float F = 1.0; //inbreeding
	int N_ind = pow(10.f,5)*(1+F); //number of individuals in population, set to maintain consistent effective number of chromosomes
	float s = gamma/(2*N_ind); //selection coefficient
	float mu = pow(10.f,-9); //per-site mutation rate
	int total_number_of_generations = 1;//36;//50;//pow(10.f,3);//
	float L = 100*2*pow(10.f,7); //number of sites
	float m = 0.00; //migration rate
	int num_pop = 1; //number of populations
	int num_iter = 50;
    int compact_rate = 35;

    G_result expectation = G(gamma,mu, L, 2.0*N_ind/(1.0+F));

	for(int i = 0; i < num_iter; i++){
		int seed1 = 0xbeeff00d + 2*i; //random number seeds
		int seed2 = 0xdecafbad - 2*i;
		sim_result * b = run_sim(const_parameter(mu), const_demography(N_ind), const_equal_migration(m,num_pop), const_selection(s), const_parameter(F), const_parameter(h), total_number_of_generations, L, num_pop, seed1, seed2, do_nothing(), do_nothing(), 0, true, sim_result(), compact_rate);
		if(i==0){ cout<< "chi-gram number of mutations:"<<endl; }
		cout<< expectation.total_SNPs << "\t" << b[0].num_mutations<< "\t" << ((b[0].num_mutations - expectation.total_SNPs)/sqrt(expectation.total_SNPs)) << endl;
		delete [] b;
	}

	delete [] expectation.g;
}
