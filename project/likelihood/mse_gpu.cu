#include "go_fish.cuh"
#include "spectrum.h"

void sfs_mse_expectation(Spectrum::MSE & mse, float * gamma, float * sel_prop, int num_categories, float theta, float num_sites){
	float pop_size = mse.Nchrom_e;
	float mu = theta/(2.f*pop_size);				//per-site mutation rate theta/2N
	float h = 0; 									//constant allele dominance (effectively ignored since F = 1)
	bool reset = true;
    for(int j = 0; j < num_categories; j++){
    	if(sel_prop[j] == 0){ continue; }
    	float sel_coeff = gamma[j]/pop_size;
    	Sim_Model::selection_constant selection(sel_coeff); 
    	//reset determines if the mse calculated replaces previous values or accumulates
    	GO_Fish::mse_SFS(mse, mu, selection, h, num_sites*sel_prop[j], reset);
    	reset = false;
    }
    			
    Spectrum::site_frequency_spectrum(mse);
}

