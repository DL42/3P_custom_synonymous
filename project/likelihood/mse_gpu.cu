#include "go_fish.cuh"
#include "spectrum.h"
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <chrono>

constexpr int SFS_size = 160;
constexpr int SFS_fold_size = ((SFS_size%2)+SFS_size)/2 + 1;
constexpr float theta = 0.01;
constexpr int pop_size = 40000;											

void run_mse_expectation(int num_iterations, int num_sites, std::vector<float> gamma, std::vector<float> sel_prop, Spectrum::MSE & mse_data_struct){
	float mu = theta/(2.f*pop_size);				//per-site mutation rate theta/2N
	float h = 0; 									//constant allele dominance (effectively ignored since F = 1)
    Sim_Model::F_mu_h_constant inbred(1.f); 								    //constant inbreeding (fully inbred)
	Sim_Model::demography_constant demography(pop_size);
	
	std::ofstream outfile;
	std::string gamma_name;
	for(int j = 0; j < gamma.size(); j++){
		gamma_name = gamma_name + "__" + std::to_string(2*gamma[j]) + "_" + std::to_string(sel_prop[j]);
	}												
	
	std::string file_name = "/home/dlawrie/Dropbox/David_Heather_SynSelection/sfs_testing/simulation_scripts/exp_mse_"+std::to_string(theta)+"/"+std::to_string(num_sites)+gamma_name+"_" + std::to_string(0) + ".txt";
	//std::string file_name = "/Users/dlawrie/Dropbox/David_Heather_SynSelection/sfs_testing/simulation_scripts/exp_mse_"+std::to_string(theta)+"/"+std::to_string(num_sites)+gamma_name+"_" + std::to_string(0) + ".txt";
	outfile.open(file_name);
	
	std::cout<<num_sites<<"\t"<<gamma_name<<std::endl;
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	start = std::chrono::high_resolution_clock::now();
	
	std::vector<float> SFS_DFE_FOLD(SFS_fold_size);
	for(int i = 0; i < num_iterations; i++){
    		bool reset = true;
    		for(int j = 0; j < gamma.size(); j++){
    			if(sel_prop[j] == 0){ continue; }
    			Sim_Model::selection_constant selection(gamma[j], demography, inbred); 
    			//reset determines if the mse calculated replaces previous values or accumulates
    			GO_Fish::mse_SFS(mse_data_struct, mu, selection, h, num_sites*sel_prop[j], reset);
    			reset = false;
    		}
    			
    		Spectrum::site_frequency_spectrum(mse_data_struct);
    		SFS_DFE_FOLD[0] = mse_data_struct.h_frequency_spectrum[0];
    		for(int k = 1; k < SFS_fold_size; k++){ 
    			if(k != SFS_size-k){ SFS_DFE_FOLD[k] = mse_data_struct.h_frequency_spectrum[k] + mse_data_struct.h_frequency_spectrum[SFS_size-k]; } 
    			else { SFS_DFE_FOLD[k] = mse_data_struct.h_frequency_spectrum[k]; } 
    		} 
	}
	
	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double,std::milli> elapsed_ms = end - start;
	std::cout<<elapsed_ms.count()/num_iterations<<std::endl;
	
	for(int k = 0; k < SFS_fold_size; k++){ if(k+1 < SFS_fold_size){ outfile<< SFS_DFE_FOLD[k] << std::endl; } else { outfile<< SFS_DFE_FOLD[k]; } }
    outfile.close();
}

int main(int argc, char **argv){ 
	std::srand(0xdecafbad);
	float num_sites = 2*pow(10,7);
	
	std::vector<std::vector<float>> gamma2_array = {{0,-2},{0,-10},{0,-25},{0,-50},{0,-100},{0,-150},{0,-200},{0,-250}};
	std::vector<std::vector<float>> sel2_prop_array = {{0.95,0.05},{0.9,0.1},{0.85,0.15},{0.8,0.2},{0.7,0.3}};
	
	std::vector<std::vector<float>> gamma3_array = {{0,-2,-25},{0,-2,-50},{0,-2,-100},{0,-2,-150},{0,-2,-200},{0,-2,-250},
													{0,-10,-25},{0,-10,-50},{0,-10,-100},{0,-10,-150},{0,-10,-200},{0,-10,-250}};
	std::vector<std::vector<float>> sel3_prop_array = {{0.85,0.1,0.05},{0.7,0.2,0.1},{0.6,0.3,0.1},{0.4,0.45,0.15}};
	
	int num_iterations = 1;
	
	Spectrum::MSE mse_data_struct(SFS_size, pop_size, 1.f);
	
	run_mse_expectation(num_iterations, num_sites, {0}, {1}, mse_data_struct);
	
	for(int j = 0; j < gamma2_array.size(); j++){
		for(int k = 0; k < sel2_prop_array.size(); k++){
			run_mse_expectation(num_iterations, num_sites, gamma2_array[j], sel2_prop_array[k], mse_data_struct);
		}
	}
		
	for(int l = 0; l < gamma3_array.size(); l++){
		for(int m = 0; m < sel3_prop_array.size(); m++){
			run_mse_expectation(num_iterations, num_sites, gamma3_array[l], sel3_prop_array[m], mse_data_struct);
		}	
	}
	
}
