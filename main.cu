#include "fw_sim_api.h"
#include "sfs.cuh"

int main(int argc, char **argv)
{

    float gamma = 0;
	float h = 0.5;
	float F = 1.0;
	int N_ind = pow(10.f,5)*(1+F); //constant population for now
	float s = gamma/(2*N_ind);
	float mu = pow(10.f,-9); //per-site mutation rate
	int total_number_of_generations = pow(10.f,5);//36;//50;//
	float L = 1*2*pow(10.f,7);
	float m = 0.00;
	int num_pop = 1;
	int seed1 = 0xbeeff00d;
	int seed2 = 0xdecafbad;

	sim_result * a = run_sim(const_mutation(mu), const_demography(N_ind), const_migration(m,num_pop), const_selection(s), const_inbreeding(F), const_dominance(h), total_number_of_generations, L, num_pop, seed1, seed2, no_sample(), 0, true);
	cout<<endl<<"final number of mutations: " << a[0].num_mutations << endl;
	int x = 60000;
	for(int i = x; i < (x+10); i++){
		cout<<"mutation freq: " << a[0].mutations_freq[i] << endl;
		cout<<"generation of mutation: " << a[0].mutations_ID[i].generation << endl;
		cout<<"thread of mutation: " << a[0].mutations_ID[i].threadID << endl;
		cout<<"population of mutation: " << a[0].mutations_ID[i].population << endl;
		cout<<"device of mutation: " << a[0].mutations_ID[i].device << endl;
		cout<<endl;
	}
	delete [] a;

	a = run_sim(const_mutation(mu), const_demography(N_ind), const_migration(m,num_pop), const_selection(s), const_inbreeding(F), const_dominance(h), total_number_of_generations, L, num_pop, seed1, seed2, no_sample(), 0, true);
	delete [] a;


    cudaEvent_t start, stop;
    float elapsedTime;
    int num_iter = 10;
    int compact_rate = 35;
    num_pop = 1;
    m = 0.0;

    total_number_of_generations = pow(10.f,3);
    L = 1*2*pow(10.f,7);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for(int i = 0; i < num_iter; i++){
		sim_result * b = run_sim(const_mutation(mu), const_demography(N_ind), const_migration(m,num_pop), const_selection(s), const_inbreeding(F), const_dominance(h), total_number_of_generations, L, num_pop, seed1, seed2, no_sample(), 0, true, sim_result(), compact_rate);
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

	cudaDeviceSynchronize();
	cudaDeviceReset();
}
