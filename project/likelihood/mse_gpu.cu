#include "go_fish.cuh"
#include "spectrum.h"

void sfs_mse_expectation(Spectrum::MSE & mse, const double * gamma, const double * h, const double F, const double * proportion, const int num_categories, const double theta, const double num_sites){
	double Nchrom_e = mse.Nchrom_e;
	double mu = theta/(2.0*Nchrom_e);				//per-site mutation rate theta/2N
	bool reset = true;
    for(int j = 0; j < num_categories; j++){
    	if(proportion[j] == 0){ continue; }
    	double sel_coeff = gamma[j]/Nchrom_e;
    	Sim_Model::selection_constant64 selection(sel_coeff); 
    	//reset determines if the mse calculated replaces previous values or accumulates
    	GO_Fish::mse_SFS(mse, mu, selection, h[j], F, num_sites*proportion[j], reset);
    	reset = false;
    }
    			
    Spectrum::site_frequency_spectrum(mse);
}

