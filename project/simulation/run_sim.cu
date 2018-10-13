#include "go_fish.cuh"
#include "spectrum.h"
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>

constexpr int SFS_size = 160;
constexpr int SFS_fold_size = ((SFS_size%2)+SFS_size)/2 + 1;
constexpr float theta = 0.035; //0.01; //
constexpr int anc_pop_size = 40000;											
#define simTOrun run_zambia_bottlegrowth_sim //run_mse_sim //

void run_mse_sim(int num_iterations, int num_sites, std::vector<float> gamma, std::vector<float> sel_prop){
	GO_Fish::allele_trajectories a;
	a.sim_input_constants.num_populations = 1; 									//1 population
	Sim_Model::F_mu_h_constant inbred(1.f); 								    //constant inbreeding (fully inbred)
	Sim_Model::demography_constant demography(anc_pop_size);						
	Sim_Model::F_mu_h_constant mutation(theta/(2*demography(0,0)));				//per-site mutation rate theta/2N
	Sim_Model::migration_constant_equal no_mig; 								//constant, 0, migration rate
	Sim_Model::F_mu_h_constant dominance(0.f); 									//constant allele dominance (effectively ignored since F = 1)
    
    a.sim_input_constants.num_generations = 10;								    //10 generations in simulation
    a.sim_input_constants.compact_interval = 55;					
    
//sample population    
    Sim_Model::demography_constant sample(SFS_size);						    //160 haploid individuals in sample
	Sim_Model::selection_constant neutral(0); 									//constant, neutral, selection coefficient
	Sim_Model::F_mu_h_constant no_mut(0); 							            //per-site mutation rate 0  	
	
	std::ofstream outfile;
	std::string gamma_name;
	for(int j = 0; j < gamma.size(); j++){
		gamma_name = gamma_name + "__" + std::to_string(2*gamma[j]) + "_" + std::to_string(sel_prop[j]);
	}												
	
	std::cout<<num_sites<<"\t"<<gamma_name<<std::endl;
	for(int i = 0; i < num_iterations; i++){
		std::string file_name = "../results/mse_"+std::to_string(theta)+"/"+std::to_string(num_sites)+gamma_name+"_" + std::to_string(i) + ".txt";
		outfile.open(file_name);
		
    	std::vector<float> SFS_DFE(SFS_size);
    	for(int k = 0; k < SFS_size; k++){ SFS_DFE[k] = 0; }
    	std::vector<float> SFS_DFE_FOLD(SFS_fold_size);
    	
    	for(int j = 0; j < gamma.size(); j++){
    		a.sim_input_constants.seed1 = std::rand(); 						    //random random number seeds
    		a.sim_input_constants.seed2 = std::rand();
    		if(sel_prop[j] == 0){ continue; }
    		a.sim_input_constants.num_sites = num_sites*sel_prop[j];			//number of sites
    		Sim_Model::selection_constant selection(gamma[j], demography, inbred); 
    		//final generation, sample alleles
    		Sim_Model::selection_piecewise<Sim_Model::selection_constant,Sim_Model::selection_constant> selection_model(selection,neutral,a.sim_input_constants.num_generations);
    		Sim_Model::demography_piecewise<Sim_Model::demography_constant,Sim_Model::demography_constant> demography_model(demography,sample,a.sim_input_constants.num_generations);
			Sim_Model::F_mu_h_piecewise<Sim_Model::F_mu_h_constant,Sim_Model::F_mu_h_constant> mutation_model(mutation,no_mut,a.sim_input_constants.num_generations);
    		GO_Fish::run_sim(a, mutation_model, demography_model, no_mig, selection_model, inbred,dominance, Sim_Model::bool_off(), Sim_Model::bool_off());
    	
    	    if(a.maximal_num_mutations() == 0){ continue; }
    		Spectrum::SFS temp;
    		Spectrum::population_frequency_histogram(temp, a, 0, 0);
    		
    		for(int k = 0; k < SFS_size; k++){ SFS_DFE[k] += temp.frequency_spectrum[k]; }
    	}
    	
    	SFS_DFE_FOLD[0] = SFS_DFE[0];
    	for(int k = 1; k < SFS_fold_size; k++){ if(k != SFS_size-k){ SFS_DFE_FOLD[k] = SFS_DFE[k] + SFS_DFE[SFS_size-k]; } else { SFS_DFE_FOLD[k] = SFS_DFE[k]; } }
    	for(int k = 0; k < SFS_fold_size; k++){ if(k+1 < SFS_fold_size){ outfile<< SFS_DFE_FOLD[k] << std::endl; } else { outfile<< SFS_DFE_FOLD[k]; } }
    	outfile.close();
	}
}

void run_zambia_bottlegrowth_sim(int num_iterations, int num_sites, std::vector<float> gamma, std::vector<float> sel_prop){
	GO_Fish::allele_trajectories a;
	a.sim_input_constants.num_populations = 1; 									//1 population
	Sim_Model::F_mu_h_constant inbred(1.f); 								    //constant inbreeding (fully inbred)
	Sim_Model::demography_constant anc_size(anc_pop_size);
	//http://www.genetics.org/content/193/1/215#T2 3rd codon pos
	int num_gen = 0.14*anc_size(0,0);
	a.sim_input_constants.num_generations = num_gen+1;	
	Sim_Model::demography_exponential_growth exp((log(0.76/0.016)/num_gen), 0.016*anc_size(0,0));		
	
	typedef Sim_Model::demography_piecewise<Sim_Model::demography_constant,Sim_Model::demography_exponential_growth> dem_model;
	typedef Sim_Model::demography_piecewise<dem_model,Sim_Model::demography_constant> sample_dem;
	
	//sample population    
    Sim_Model::demography_constant sample(SFS_size);						    //160 haploid individuals in sample
	Sim_Model::selection_constant neutral(0); 									//constant, neutral, selection coefficient
	Sim_Model::F_mu_h_constant no_mut(0); 							            //per-site mutation rate 0  
	
	dem_model bottlegrowth(anc_size,exp,1);
    sample_dem demography_model(bottlegrowth,sample,a.sim_input_constants.num_generations);	
	
	Sim_Model::migration_constant_equal no_mig; 								//constant, 0, migration rate
	Sim_Model::F_mu_h_constant dominance(0.f); 									//constant allele dominance (effectively ignored since F = 1)
    
   	Sim_Model::F_mu_h_constant mutation(1.812529864f*theta/(2*anc_pop_size));				//per-site mutation rate theta/2N				
    Sim_Model::F_mu_h_piecewise<Sim_Model::F_mu_h_constant,Sim_Model::F_mu_h_constant> mutation_model(mutation,no_mut,a.sim_input_constants.num_generations);

    a.sim_input_constants.compact_interval = 55;					
    	
	std::ofstream outfile;
	std::string gamma_name;
	for(int j = 0; j < gamma.size(); j++){
		gamma_name = gamma_name + "__" + std::to_string(2*gamma[j]) + "_" + std::to_string(sel_prop[j]);
	}
	
	std::cout<<num_sites<<"\t"<<gamma_name<<std::endl;
	for(int i = 0; i < num_iterations; i++){
		std::string file_name = "../results/zambia_bottlegrowth_"+std::to_string(theta)+"/"+std::to_string(num_sites)+gamma_name+"_" + std::to_string(i) + ".txt";
		outfile.open(file_name);
	
    	std::vector<float> SFS_DFE(SFS_size);
    	for(int k = 0; k < SFS_size; k++){ SFS_DFE[k] = 0; }
    	std::vector<float> SFS_DFE_FOLD(SFS_fold_size);
    	
    	for(int j = 0; j < gamma.size(); j++){
    		a.sim_input_constants.seed1 = std::rand(); 							    //random random number seeds
    		a.sim_input_constants.seed2 = std::rand();
    		if(sel_prop[j] == 0){ continue; }
    		a.sim_input_constants.num_sites = num_sites*sel_prop[j];			//number of sites
    		Sim_Model::selection_constant selection(1.812529864f*gamma[j]/anc_pop_size); 

    		Sim_Model::selection_piecewise<Sim_Model::selection_constant,Sim_Model::selection_constant> selection_model(selection,neutral,a.sim_input_constants.num_generations);
    		GO_Fish::run_sim(a, mutation_model, demography_model, no_mig, selection_model, inbred,dominance, Sim_Model::bool_off(), Sim_Model::bool_off());
			if(a.maximal_num_mutations() == 0){ continue; }
    		Spectrum::SFS temp;
    		Spectrum::population_frequency_histogram(temp, a, 0, 0);
    		for(int k = 0; k < SFS_size; k++){ SFS_DFE[k] += temp.frequency_spectrum[k]; }
    	}
    	SFS_DFE_FOLD[0] = SFS_DFE[0];
    	for(int k = 1; k < SFS_fold_size; k++){ if(k != SFS_size-k){ SFS_DFE_FOLD[k] = SFS_DFE[k] + SFS_DFE[SFS_size-k]; } else { SFS_DFE_FOLD[k] = SFS_DFE[k]; } }
    	for(int k = 0; k < SFS_fold_size; k++){ if(k+1 < SFS_fold_size){ outfile<< SFS_DFE_FOLD[k] << std::endl; } else { outfile<< SFS_DFE_FOLD[k]; } }    	
		outfile.close();
	}
}

int main(int argc, char **argv){ 
	std::srand(0xdecafbad);
	std::vector<int> num_sites_array = {int(100*pow(10,3)),int(200*pow(10,3)),int(400*pow(10,3)),int(800*pow(10,3)),int(1600*pow(10,3)),int(3200*pow(10,3))};
	
	std::vector<std::vector<float>> gamma2_array = {{0,-2},{0,-10},{0,-25},{0,-50},{0,-100},{0,-150},{0,-200},{0,-250}};
	std::vector<std::vector<float>> sel2_prop_array = {{0.95,0.05},{0.9,0.1},{0.85,0.15},{0.8,0.2},{0.7,0.3}};
	
	std::vector<std::vector<float>> gamma3_array = {{0,-2,-25},{0,-2,-50},{0,-2,-100},{0,-2,-150},{0,-2,-200},{0,-2,-250},
													{0,-10,-25},{0,-10,-50},{0,-10,-100},{0,-10,-150},{0,-10,-200},{0,-10,-250}};
	std::vector<std::vector<float>> sel3_prop_array = {{0.85,0.1,0.05},{0.7,0.2,0.1},{0.6,0.3,0.1},{0.4,0.45,0.15}};
	
	int num_iterations = 200;
	
	for(int i = 0; i < num_sites_array.size(); i++){
		simTOrun(num_iterations, num_sites_array[i], {0}, {1});
		for(int j = 0; j < gamma2_array.size(); j++){
			for(int k = 0; k < sel2_prop_array.size(); k++){
				simTOrun(num_iterations, num_sites_array[i], gamma2_array[j], sel2_prop_array[k]);
			}
		}
		
	    for(int l = 0; l < gamma3_array.size(); l++){
			for(int m = 0; m < sel3_prop_array.size(); m++){
				simTOrun(num_iterations, num_sites_array[i], gamma3_array[l], sel3_prop_array[m]);
			}
		}	
	}
}
