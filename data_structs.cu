#include "data_structs.h"

sim_result::sim_result(): num_populations(0), num_mutations(0), num_sites(0), total_generations(0) { mutations_freq = NULL; mutations_ID = NULL; extinct = NULL; }
sim_result::~sim_result(){ if(mutations_freq){ cudaCheckErrors(cudaFreeHost(mutations_freq),-1,-1); } if(mutations_ID){ cudaCheckErrors(cudaFreeHost(mutations_ID),-1,-1); } if(extinct){ delete [] extinct; } }
