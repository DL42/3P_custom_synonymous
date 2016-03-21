#include "sim_results.h"

sim_result::sim_result(): num_populations(0), num_mutations(0), num_sites(0), total_generations(0) { mutations_freq = NULL; mutations_ID = NULL; extinct = NULL; }
sim_result::~sim_result(){ if(mutations_freq){ cudaFreeHost(mutations_freq); } if(mutations_ID){ cudaFreeHost(mutations_ID); } if(extinct){ delete [] extinct; } }
