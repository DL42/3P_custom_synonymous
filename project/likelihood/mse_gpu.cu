#include "go_fish.cuh"
#include "spectrum.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void sfs_mse_expectation(py::array_t<float> SFS_DFE, py::array_t<float> gamma, py::array_t<float> sel_prop, float theta, float num_sites, Spectrum::MSE & mse_data_struct){
	float pop_size = mse_data_struct.Nchrom_e;
	float mu = theta/(2.f*pop_size);				//per-site mutation rate theta/2N
	float h = 0; 									//constant allele dominance (effectively ignored since F = 1)
    int SFS_size = SFS_DFE.size();										
	bool reset = true;
    for(int j = 0; j < gamma.size(); j++){
    	if(sel_prop.at(j) == 0){ continue; }
    	float sel_coeff = gamma.at(j)/pop_size;
    	Sim_Model::selection_constant selection(sel_coeff); 
    	//reset determines if the mse calculated replaces previous values or accumulates
    	GO_Fish::mse_SFS(mse_data_struct, mu, selection, h, num_sites*sel_prop.at(j), reset);
    	reset = false;
    }
    			
    Spectrum::site_frequency_spectrum(mse_data_struct);
    auto ptr = SFS_DFE.mutable_data(0);
    for(int k = 0; k < SFS_size; k++){ ptr[k] = mse_data_struct.h_frequency_spectrum[k]; }
}

PYBIND11_MODULE(mse_gpu, m)
{	
	py::class_<Spectrum::MSE>(m, "MSE")
        .def(py::init<const int, const int, const float, int>(),
             py::arg("sample_size"), 
             py::arg("population_size"),
             py::arg("inbreeding"),
             py::arg("cuda_device")=-1);

	m.def("sfs_mse_expectation", &sfs_mse_expectation);
}