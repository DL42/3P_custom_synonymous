#include "go_fish.cuh"
#include "spectrum.h"
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>

constexpr int SFS_size = 160;
constexpr int SFS_fold_size = ((SFS_size%2)+SFS_size)/2 + 1;
constexpr float theta = 0.01; //0.035; //
constexpr int anc_pop_size = 40000;	
constexpr float num_sites = 20000000.f;
constexpr int num_iterations = 1; //only for bottlegrowth
#define simTOrun run_all_mse //run_all_bottlegrowth //

///////////////////////////////////////// bottlegrowth

void run_zambia_bottlegrowth_expectation(int num_iterations, float num_sites, std::vector<float> gamma, std::vector<float> sel_prop){
	GO_Fish::allele_trajectories a;
	a.sim_input_constants.num_populations = 1; 									//1 population
	Sim_Model::F_mu_h_constant inbred(1.f); 								    //constant inbreeding (fully inbred)
	Sim_Model::demography_constant anc_size(anc_pop_size);
	//http://www.genetics.org/content/193/1/215#T2 3rd codon pos
	int num_gen = 0.14*anc_size(0,0);
	a.sim_input_constants.num_generations = num_gen;	
	Sim_Model::demography_exponential_growth exp((log(0.76/0.016)/num_gen), 0.016*anc_size(0,0));		
	
	typedef Sim_Model::demography_piecewise<Sim_Model::demography_constant,Sim_Model::demography_exponential_growth> dem_model;
	dem_model bottlegrowth(anc_size,exp,1);
	
	Sim_Model::migration_constant_equal no_mig; 								//constant, 0, migration rate
	Sim_Model::F_mu_h_constant dominance(0.f); 									//constant allele dominance (effectively ignored since F = 1)
    
   	Sim_Model::F_mu_h_constant mutation(1.812529864f*theta/(2*anc_pop_size));				//per-site mutation rate theta/2N //constant determined by difference between ancestral theta and final theta estimated by ML				

    a.sim_input_constants.compact_interval = 35;					
    	
	std::ofstream outfile;
	std::string gamma_name;
	for(int j = 0; j < gamma.size(); j++){
		gamma_name = gamma_name + "__" + std::to_string(2*gamma[j]) + "_" + std::to_string(sel_prop[j]);
	}
	
	std::cout<<int(num_sites)<<"\t"<<gamma_name<<std::endl;
	
	for(int i = 0; i < num_iterations; i++){
		std::string file_name = "../results/exp_zambia_bottlegrowth_"+std::to_string(theta)+"/"+std::to_string(int(num_sites))+gamma_name+"_" + std::to_string(i) + ".txt";
	
		outfile.open(file_name);
	
    	std::vector<float> SFS_DFE(SFS_size);
    	for(int k = 0; k < SFS_size; k++){ SFS_DFE[k] = 0; }
    	std::vector<float> SFS_DFE_FOLD(SFS_fold_size);
    	
    	for(int j = 0; j < gamma.size(); j++){
    		a.sim_input_constants.seed1 = std::rand(); 							    //random random number seeds
    		a.sim_input_constants.seed2 = std::rand();
    		if(sel_prop[j] == 0){ continue; }
    		a.sim_input_constants.num_sites = num_sites*sel_prop[j];			//number of sites
    		Sim_Model::selection_constant selection(1.812529864f*gamma[j]/anc_pop_size); //constant determined by difference between ancestral theta and final theta estimated by ML

    		GO_Fish::run_sim(a, mutation, bottlegrowth, no_mig, selection, inbred, dominance, Sim_Model::bool_off(), Sim_Model::bool_off());
			if(a.maximal_num_mutations() == 0){ continue; }
    		Spectrum::SFS temp;
    		Spectrum::site_frequency_spectrum(temp, a, 0, 0, SFS_size);
    		for(int k = 0; k < SFS_size; k++){ SFS_DFE[k] += temp.frequency_spectrum[k]; }
    	}
    	SFS_DFE_FOLD[0] = SFS_DFE[0];
    	for(int k = 1; k < SFS_fold_size; k++){ if(k != SFS_size-k){ SFS_DFE_FOLD[k] = SFS_DFE[k] + SFS_DFE[SFS_size-k]; } else { SFS_DFE_FOLD[k] = SFS_DFE[k]; } }
    	for(int k = 0; k < SFS_fold_size; k++){ if(k+1 < SFS_fold_size){ outfile<< SFS_DFE_FOLD[k] << std::endl; } else { outfile<< SFS_DFE_FOLD[k]; } }    	
		outfile.close();
	}
}

void run_all_bottlegrowth(std::vector<std::vector<float>> & gamma2_array, std::vector<std::vector<float>> & sel2_prop_array, std::vector<std::vector<float>> gamma3_array, std::vector<std::vector<float>> & sel3_prop_array){
	run_zambia_bottlegrowth_expectation(num_iterations, num_sites, {0}, {1});
	
	for(int j = 0; j < gamma2_array.size(); j++){
		for(int k = 0; k < sel2_prop_array.size(); k++){
			run_zambia_bottlegrowth_expectation(num_iterations, num_sites, gamma2_array[j], sel2_prop_array[k]);
		}
	}
		
	for(int l = 0; l < gamma3_array.size(); l++){
		for(int m = 0; m < sel3_prop_array.size(); m++){
			run_zambia_bottlegrowth_expectation(num_iterations, num_sites, gamma3_array[l], sel3_prop_array[m]);
		}	
	}
}

/////////////////////////////////////////	mutation-selection equilibrium
										
void run_mse_expectation(std::vector<float> gamma, std::vector<float> sel_prop, Spectrum::MSE & mse_data_struct){
	float mu = theta/(2.f*anc_pop_size);				//per-site mutation rate theta/2N
	float h = 0; 									//constant allele dominance (effectively ignored since F = 1)
	
	std::ofstream outfile;
	std::string gamma_name;
	for(int j = 0; j < gamma.size(); j++){
		gamma_name = gamma_name + "__" + std::to_string(2*gamma[j]) + "_" + std::to_string(sel_prop[j]);
	}												
	
	std::string file_name = "../results/exp_mse_"+std::to_string(theta)+"/"+std::to_string(int(num_sites))+gamma_name+"_" + std::to_string(0) + ".txt";
	outfile.open(file_name);
	
	std::cout<<int(num_sites)<<"\t"<<gamma_name<<std::endl;
	
	std::vector<float> SFS_DFE_FOLD(SFS_fold_size);
	bool reset = true;
    for(int j = 0; j < gamma.size(); j++){
    	if(sel_prop[j] == 0){ continue; }
    	float sel_coeff = gamma[j]/anc_pop_size;
    	Sim_Model::selection_constant selection(sel_coeff); 
    	//reset determines if the mse calculated replaces previous values or accumulates
    	GO_Fish::mse_SFS(mse_data_struct, mu, selection, h, 1.f, num_sites*sel_prop[j], reset);
    	reset = false;
    }
    			
    Spectrum::site_frequency_spectrum(mse_data_struct);
    SFS_DFE_FOLD[0] = mse_data_struct.h_frequency_spectrum[0];
    for(int k = 1; k < SFS_fold_size; k++){ 
    	if(k != SFS_size-k){ SFS_DFE_FOLD[k] = mse_data_struct.h_frequency_spectrum[k] + mse_data_struct.h_frequency_spectrum[SFS_size-k]; } 
    	else { SFS_DFE_FOLD[k] = mse_data_struct.h_frequency_spectrum[k]; } 
    } 
	
	for(int k = 0; k < SFS_fold_size; k++){ if(k+1 < SFS_fold_size){ outfile<< SFS_DFE_FOLD[k] << std::endl; } else { outfile<< SFS_DFE_FOLD[k]; } }
    outfile.close();
}

void run_all_mse(std::vector<std::vector<float>> & gamma2_array, std::vector<std::vector<float>> & sel2_prop_array, std::vector<std::vector<float>> gamma3_array, std::vector<std::vector<float>> & sel3_prop_array){
	Spectrum::MSE mse_data_struct(SFS_size, anc_pop_size, true, true);
	
	run_mse_expectation({0}, {1}, mse_data_struct);
	
	for(int j = 0; j < gamma2_array.size(); j++){
		for(int k = 0; k < sel2_prop_array.size(); k++){
			run_mse_expectation(gamma2_array[j], sel2_prop_array[k], mse_data_struct);
		}
	}
		
	for(int l = 0; l < gamma3_array.size(); l++){
		for(int m = 0; m < sel3_prop_array.size(); m++){
			run_mse_expectation(gamma3_array[l], sel3_prop_array[m], mse_data_struct);
		}	
	}
}

/////////////////////////////////////////	main

int main(int argc, char **argv){ 
	std::vector<std::vector<float>> gamma2_array = {{0,-2},{0,-10},{0,-25},{0,-50},{0,-100},{0,-150},{0,-200},{0,-250}};
	std::vector<std::vector<float>> sel2_prop_array = {{0.95,0.05},{0.9,0.1},{0.85,0.15},{0.8,0.2},{0.7,0.3}};	
	std::vector<std::vector<float>> gamma3_array = {{0,-2,-25},{0,-2,-50},{0,-2,-100},{0,-2,-150},{0,-2,-200},{0,-2,-250},
													{0,-10,-25},{0,-10,-50},{0,-10,-100},{0,-10,-150},{0,-10,-200},{0,-10,-250}};
	std::vector<std::vector<float>> sel3_prop_array = {{0.85,0.1,0.05},{0.7,0.2,0.1},{0.6,0.3,0.1},{0.4,0.45,0.15}};

	simTOrun(gamma2_array, sel2_prop_array, gamma3_array, sel3_prop_array);
}
