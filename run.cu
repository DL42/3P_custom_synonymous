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
    float gamma = 0;
	float h = 0.5;
	float F = 1.0;
	int N_ind = pow(10.f,5)*(1+F);
	float s = gamma/(2*N_ind);
	float mu = pow(10.f,-9); //per-site mutation rate
	int total_number_of_generations = pow(10.f,5);//36;//50;//
	float L = 2*pow(10.f,7);
	float m = 0.00;
	int num_pop = 1;
	int seed1 = 0xbeeff00d;
	int seed2 = 0xdecafbad;
	bool printSFS = true;

	//----- warm up GPU -----
	sim_result * a = run_sim(const_mutation(mu), const_demography(N_ind), const_migration(m,num_pop), const_selection(s), const_inbreeding(F), const_dominance(h), total_number_of_generations, L, num_pop, seed1, seed2, no_preserve(), no_sample(), 0, true);
	cout<<endl<<"final number of mutations: " << a[0].num_mutations << endl;

	//----- print allele counts x to x+y of warm up GPU scenario -----
	int start_index = 0;
	int print_num = 50;
	if(printSFS){
		SFS::sfs mySFS = SFS::site_frequency_spectrum(a[0],0);
		cout<< "allele count\t# mutations"<< endl;
		for(int printIndex = start_index; printIndex < min((mySFS.num_samples[0]-start_index),start_index+print_num); printIndex++){ cout<< (printIndex) << "\t" << mySFS.frequency_spectrum[printIndex] <<endl;}
	}
	//----- end print allele counts x to x+y of warm up GPU scenario -----
	delete [] a;

	a = run_sim(const_mutation(mu), const_demography(N_ind), const_migration(m,num_pop), const_selection(s), const_inbreeding(F), const_dominance(h), total_number_of_generations, L, num_pop, seed1, seed2, no_preserve(), no_sample(), 0, true);
	delete [] a;
	//----- end warm up GPU -----

    cudaEvent_t start, stop;
    float elapsedTime;
    int num_iter = 10;
    int compact_rate = 35;
    gamma = 0;
    N_ind = pow(10.f,5)*(1+F);
    s = gamma/(2*N_ind);
    total_number_of_generations = pow(10.f,3);
    L = 1*2*pow(10.f,7);
    num_pop = 1;
    m = 0.0;

    //----- speed test -----
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for(int i = 0; i < num_iter; i++){
		sim_result * b = run_sim(const_mutation(mu), const_demography(N_ind), const_migration(m,num_pop), const_selection(s), const_inbreeding(F), const_dominance(h), total_number_of_generations, L, num_pop, seed1, seed2, no_preserve(), no_sample(), 0, true, sim_result(), compact_rate);
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
