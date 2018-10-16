#include "go_fish.cuh"
#include "spectrum.h"

void sfs_mse_expectation(Spectrum::MSE & mse, const float * gamma, const float * h, const float F, const float * proportion, const int num_categories, const float theta, const float num_sites){
	float Nchrom_e = mse.Nchrom_e;
	float mu = theta/(2.f*Nchrom_e);				//per-site mutation rate theta/2N
	bool reset = true;
    for(int j = 0; j < num_categories; j++){
    	if(proportion[j] == 0){ continue; }
    	float sel_coeff = gamma[j]/Nchrom_e;
    	Sim_Model::selection_constant selection(sel_coeff); 
    	//reset determines if the mse calculated replaces previous values or accumulates
    	GO_Fish::mse_SFS(mse, mu, selection, h[j], F, num_sites*proportion[j], reset);
    	reset = false;
    }
    			
    Spectrum::site_frequency_spectrum(mse);
}

